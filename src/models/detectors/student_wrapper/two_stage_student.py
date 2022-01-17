import torch
from mmdet.core import multi_apply
from mmdet.core import bbox_overlaps
from src.utils.debug_utils import Timer
from src.utils.structure_utils import dict_concat, dict_sum, weighted_loss


from src.models.utils.semi_wrapper import (
    get_roi_loss,
    get_roi_prediction,
    split_roi_prediction,
    split_rpn_output,
)
from ..semi_base import Student


class TwoStageStudent(Student):
    def __init__(self, detector, train_cfg=None):
        super().__init__(detector, train_cfg)

    def get_prediction(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        proposals=None,
        gt_masks=None,
        gt_bboxes_ignore=None,
        **kwargs
    ):
        x = self.detector.extract_feat(img)
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                "rpn_proposal", self.detector.test_cfg.rpn
            )
            rpn_output = self.detector.rpn_head(x)
            proposals = self.detector.rpn_head.get_bboxes(
                *rpn_output, img_metas, proposal_cfg
            )
        else:
            rpn_output = None

        roi_output = get_roi_prediction(
            self.detector.roi_head,
            x,
            proposals,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_masks,
            gt_bboxes_ignore,
            **kwargs
        )

        return rpn_output, roi_output

    def get_rpn_loss(self, rpn_output, gt_bboxes, img_metas):
        if self.detector.with_rpn:
            loss_inputs = tuple(rpn_output) + (gt_bboxes, img_metas)
            return self.detector.rpn_head.loss(*loss_inputs)
        return {}

    def parallel_learn(self, labeled, unlabeled, teacher):
        if unlabeled is not None:
            with Timer("techer prediction transform"):
                pseudo_gt = teacher.deliver(unlabeled["img"], unlabeled["img_metas"])
                gt_proposals = {
                    "bboxes": unlabeled["gt_bboxes"],
                    "img_metas": unlabeled["img_metas"],
                }
                gt_scores = teacher.rate(**gt_proposals)
                bboxes, labels = multi_apply(
                    self.combine,
                    unlabeled["gt_bboxes"],
                    unlabeled["gt_labels"],
                    gt_scores,
                    [bbox[:, :4] for bbox in pseudo_gt["gt_bboxes"]],
                    pseudo_gt["gt_labels"],
                )
                unlabeled.update(gt_bboxes=bboxes, gt_labels=labels)
        loss = {}
        if labeled is None:
            with Timer("Get student prediction"):
                rpn_pred, roi_pred = self.get_prediction(**unlabeled)
                unlabeled_rpn_loss = self.get_rpn_loss(
                    rpn_pred, unlabeled["gt_bboxes"], unlabeled["img_metas"]
                )
                unlabeled_roi_loss = get_roi_loss(
                    self.detector.roi_head,
                    roi_pred,
                    unlabeled["img_metas"],
                    teacher=teacher,
                )
            loss.update(
                weighted_loss(
                    unlabeled_rpn_loss,
                    self.train_cfg.get("unsup_weight", 2.0),
                    teacher.ignore_branch,
                )
            )
            loss.update(
                weighted_loss(
                    unlabeled_roi_loss,
                    self.train_cfg.get("unsup_weight", 2.0),
                    teacher.ignore_branch,
                )
            )
        elif unlabeled is None:
            with Timer("Get student prediction"):
                rpn_pred, roi_pred = self.get_prediction(**labeled)
                labeled_rpn_loss = self.get_rpn_loss(
                    rpn_pred, labeled["gt_bboxes"], labeled["img_metas"]
                )
                labeled_roi_loss = get_roi_loss(
                    self.detector.roi_head, roi_pred, labeled["img_metas"],
                )
            loss.update(labeled_rpn_loss)
            loss.update(labeled_roi_loss)
        else:
            labeled_sample_num, unlabeled_sample_num = (
                len(labeled["img"]),
                len(unlabeled["img"]),
            )
            with Timer("Get student prediction"):
                rpn_pred, roi_pred = self.get_prediction(
                    **dict_concat([labeled, unlabeled])
                )

            with Timer("Get student rpn loss"):
                labeled_rpn_pred, unlabeled_rpn_pred = split_rpn_output(
                    rpn_pred, [labeled_sample_num, unlabeled_sample_num]
                )

                labeled_rpn_loss = self.get_rpn_loss(
                    labeled_rpn_pred, labeled["gt_bboxes"], labeled["img_metas"]
                )
                unlabeled_rpn_loss = self.get_rpn_loss(
                    unlabeled_rpn_pred, unlabeled["gt_bboxes"], unlabeled["img_metas"]
                )
            with Timer("Get student rcnn loss"):
                labeled_roi_pred, unlabeled_roi_pred = split_roi_prediction(
                    self.detector.roi_head,
                    roi_pred,
                    [labeled_sample_num, unlabeled_sample_num],
                )

                labeled_roi_loss = get_roi_loss(
                    self.detector.roi_head, labeled_roi_pred, labeled["img_metas"]
                )
                unlabeled_roi_loss = get_roi_loss(
                    self.detector.roi_head,
                    unlabeled_roi_pred,
                    unlabeled["img_metas"],
                    teacher=teacher,
                )

            loss.update(labeled_rpn_loss)
            loss.update(labeled_roi_loss)

            unlabeled_loss = {}
            unlabeled_loss.update(
                weighted_loss(
                    unlabeled_rpn_loss,
                    self.train_cfg.get("unsup_weight", 2.0),
                    teacher.ignore_branch,
                )
            )
            unlabeled_loss.update(
                weighted_loss(
                    unlabeled_roi_loss,
                    self.train_cfg.get("unsup_weight", 2.0),
                    teacher.ignore_branch,
                )
            )
            loss = dict_sum(loss, unlabeled_loss)
        return loss

    def combine(self, bbox_gt, label_gt, score_gt, bbox_noise, label_noise):
        score_thr = self.train_cfg.get("gt_score_thr", 0.9)
        flags = score_gt[torch.arange(len(score_gt)), label_gt] > score_thr
        bbox_gt = bbox_gt[flags]
        label_gt = label_gt[flags]

        iou = bbox_overlaps(bbox_gt, bbox_noise)
        if iou.numel() > 0:
            matched = iou.float().max(dim=1)[0] > self.train_cfg.get("iou_thr", 0.5)
        else:
            matched = torch.zeros_like(label_gt, dtype=torch.bool)

        bbox_select = torch.cat([bbox_noise, bbox_gt[~matched]])
        label_select = torch.cat([label_noise, label_gt[~matched]])

        return bbox_select, label_select
