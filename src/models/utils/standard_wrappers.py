import numpy as np
import torch
from torch.nn import functional as F
from mmdet.core import bbox2roi, bbox_overlaps, roi2bbox
from src.utils.structure_utils import dict_concat


def get_bbox_confidence(self, x, img_metas, proposal_list, **kwargs):
    assert self.with_bbox, "Bbox head must be implemented."
    num_proposals_per_img = tuple(len(proposals) for proposals in proposal_list)
    rois = bbox2roi(proposal_list)
    bbox_results = self._bbox_forward(x, rois)
    cls_score = bbox_results["cls_score"].softmax(dim=-1)
    cls_score = cls_score.split(num_proposals_per_img, 0)
    return cls_score


def get_roi_prediction(
    self,
    x,
    proposal_list,
    img_metas,
    gt_bboxes,
    gt_labels,
    gt_masks,
    gt_bboxes_ignore=None,
    **kwargs,
):
    if self.with_bbox or self.with_mask:
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x],
            )
            sampling_results.append(sampling_result)

        prediction = {}

        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            if kwargs.get("save_assign_gt_inds", False):
                pos_inds = [res.pos_inds for res in sampling_results]
                prediction["pos_inds"] = pos_inds
                pos_assigned_gt_inds = [
                    res.pos_assigned_gt_inds for res in sampling_results
                ]
                prediction["pos_assigned_gt_inds"] = pos_assigned_gt_inds
            bbox_results = self._bbox_forward(x, rois)
            bbox_targets = self.bbox_head.get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg
            )

            prediction["rois"] = rois
            prediction["bbox_results"] = bbox_results
            prediction["bbox_targets"] = bbox_targets
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
                mask_results = self._mask_forward(x, pos_rois)
            else:
                pos_inds = []
                device = bbox_results["bbox_feats"].device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0], device=device, dtype=torch.uint8,
                        )
                    )
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0], device=device, dtype=torch.uint8,
                        )
                    )
                pos_inds = torch.cat(pos_inds)

                mask_results = self._mask_forward(
                    x, pos_inds=pos_inds, bbox_feats=bbox_results["bbox_feats"]
                )
            mask_targets = self.mask_head.get_targets(
                sampling_results, gt_masks, self.train_cfg
            )
            pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
            prediction["mask_num_per_img"] = [
                len(res.pos_bboxes) for res in sampling_results
            ]
            prediction["mask_targets"] = mask_targets
            prediction["pos_labels"] = pos_labels
            prediction["mask_results"] = mask_results
        return prediction
    else:
        return None


def split_roi_prediction(self, prediction, splits):

    interval = [0] + np.cumsum(splits).tolist()
    if self.with_bbox:
        proposal_list = roi2bbox(prediction["rois"])
        # split rois
        proposal_lists = [
            proposal_list[interval[i] : interval[i + 1]] for i in range(len(splits))
        ]
        rois_list = [bbox2roi(p) for p in proposal_lists]
        chunk_sizes = [len(rois) for rois in rois_list]

        bbox_results = {
            k: torch.split(v, chunk_sizes)
            for k, v in prediction["bbox_results"].items()
        }
        bbox_result_list = [
            {k: v[i] for k, v in bbox_results.items()} for i in range(len(splits))
        ]

        bbox_targets = [torch.split(v, chunk_sizes) for v in prediction["bbox_targets"]]
        bbox_target_list = [[bt[i] for bt in bbox_targets] for i in range(len(splits))]
        prediction_chunks = [
            dict(rois=r, bbox_results=br, bbox_targets=bt)
            for r, br, bt in zip(rois_list, bbox_result_list, bbox_target_list)
        ]
    if self.with_mask:
        mask_num_per_img = prediction["mask_num_per_img"]
        mask_nums = [
            mask_num_per_img[interval[i] : interval[i + 1]] for i in range(len(splits))
        ]
        chunk_sizes = [sum(m) for m in mask_nums]
        mask_targets = torch.split(prediction["mask_targets"], chunk_sizes)
        pos_labels = torch.split(prediction["pos_labels"], chunk_sizes)
        mask_results = {
            k: torch.split(v, chunk_sizes)
            for k, v in prediction["mask_results"].items()
        }
        mask_results = [
            {k: v[i] for k, v in mask_results.items()} for i in range(len(splits))
        ]

        for i, p in enumerate(prediction_chunks):
            p.update(
                dict(
                    mask_results=mask_results[i],
                    mask_targets=mask_targets[i],
                    pos_labels=pos_labels[i],
                )
            )

    return prediction_chunks


def get_roi_loss(self, pred, img_metas, teacher=None, student=None, **kwargs):
    loss = dict()
    if self.with_bbox:
        bbox_results = pred["bbox_results"]
        bbox_targets = list(pred["bbox_targets"])
        rois = pred["rois"]
        with torch.no_grad():
            if (teacher is not None) and teacher.train_cfg.get(
                "with_soft_teacher", True
            ):
                unsup_flag = ["unsup" in meta["tag"] for meta in img_metas]
                if sum(unsup_flag) > 0:
                    label_weights = bbox_targets[1].detach().clone()
                    proposal_list = roi2bbox(rois)
                    label_weights = list(
                        torch.split(
                            label_weights, [len(p) for p in proposal_list], dim=0
                        )
                    )
                    label_list = torch.split(
                        bbox_targets[0], [len(p) for p in proposal_list], dim=0
                    )
                    unsup_proposals = [
                        {"bboxes": [proposal], "img_metas": [meta]}
                        for proposal, flag, meta in zip(
                            proposal_list, unsup_flag, img_metas,
                        )
                        if flag
                    ]
                    unsup_proposals = dict_concat(unsup_proposals)
                    rated_weights = teacher.rate(**unsup_proposals)
                    unsup_inds = 0
                    for i, flag in enumerate(unsup_flag):
                        if flag:
                            if (
                                teacher.train_cfg.get("rate_method", "background")
                                == "background"
                            ):
                                neg_inds = label_list[i] == self.bbox_head.num_classes
                                label_weights[i][neg_inds] = (
                                    label_weights[i][neg_inds]
                                    * rated_weights[unsup_inds][:, -1][neg_inds]
                                )
                            elif (
                                teacher.train_cfg.get("rate_method", "background")
                                == "per_class"
                            ):
                                label_weights[i] = (
                                    label_weights[i]
                                    * rated_weights[unsup_inds][
                                        torch.arange(len(rated_weights[unsup_inds])),
                                        label_list[i],
                                    ]
                                )
                            else:
                                raise NotImplementedError()
                            unsup_inds += 1
                    label_weights = torch.cat(label_weights)
                    bbox_targets[1] = (
                        label_weights.shape[0]
                        * label_weights
                        / max(label_weights.sum(), 1)
                    )
        loss.update(
            self.bbox_head.loss(
                bbox_results["cls_score"],
                bbox_results["bbox_pred"],
                rois,
                *bbox_targets,
            )
        )
    return loss


def compute_kl_loss(logits, target_prob, bg_weight=1.0, fg_weight=1.0, weight=1.0):
    C = logits.shape[-1]
    class_weight = logits.new_ones(1, C)
    class_weight[:, -1] = class_weight[:, -1] * bg_weight
    class_weight[:, :-1] = class_weight[:, :-1] * fg_weight

    loss = -1 * class_weight * target_prob * F.log_softmax(logits, dim=-1)
    loss = loss.sum(dim=-1).mean()
    return weight * loss
