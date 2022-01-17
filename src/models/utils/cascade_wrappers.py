import torch
from mmdet.core import bbox2roi, roi2bbox
from src.utils.structure_utils import dict_concat

from .standard_wrappers import split_roi_prediction as _base_split


def get_bbox_confidence(self, x, img_metas, proposal_list, **kwargs):
    assert self.with_bbox, "Bbox head must be implemented."
    num_imgs = len(proposal_list)
    rois = bbox2roi(proposal_list)

    stage = kwargs.get("stage", None)

    if stage is None:
        ms_scores = []
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)
            # split batch bbox prediction back to each image
            cls_score = bbox_results["cls_score"]
            bbox_pred = bbox_results["bbox_pred"]
            num_proposals_per_img = tuple(len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img
                )
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat(
                    [
                        self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label[j], bbox_pred[j], img_metas[j]
                        )
                        for j in range(num_imgs)
                    ]
                )

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        return [c.softmax(dim=-1) for c in cls_score]

    else:
        num_proposals_per_img = tuple(len(proposals) for proposals in proposal_list)
        bbox_results = self._bbox_forward(stage, x, rois)
        # split batch bbox prediction back to each image
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
    prediction = {}
    for i in range(self.num_stages):
        self.current_stage = i
        rcnn_train_cfg = self.train_cfg[i]

        # assign gts and sample proposals
        sampling_results = []
        if self.with_bbox or self.with_mask:
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j], gt_labels[j],
                )
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x],
                )
                sampling_results.append(sampling_result)

        _prediction = {}
        # bbox head forward
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(i, x, rois)
        bbox_targets = self.bbox_head[i].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg
        )
        _prediction["rois"] = rois
        _prediction["bbox_results"] = bbox_results
        _prediction["bbox_targets"] = bbox_targets
        # mask head forward and loss
        if self.with_mask:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(i, x, pos_rois)
            mask_targets = self.mask_head[i].get_targets(
                sampling_results, gt_masks, rcnn_train_cfg
            )
            pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
            _prediction["mask_targets"] = mask_targets
            _prediction["pos_labels"] = pos_labels
            _prediction["mask_results"] = mask_results
            _prediction["mask_num_per_img"] = [
                len(res.pos_bboxes) for res in sampling_results
            ]
        prediction[f"s{i}"] = _prediction

        # refine bboxes
        if i < self.num_stages - 1:
            pos_is_gts = [res.pos_is_gt for res in sampling_results]
            # bbox_targets is a tuple
            roi_labels = bbox_targets[0]
            with torch.no_grad():
                roi_labels = torch.where(
                    roi_labels == self.bbox_head[i].num_classes,
                    bbox_results["cls_score"][:, :-1].argmax(1),
                    roi_labels,
                )
                proposal_list = self.bbox_head[i].refine_bboxes(
                    _prediction["rois"],
                    roi_labels,
                    bbox_results["bbox_pred"],
                    pos_is_gts,
                    img_metas,
                )
    if len(prediction) == 0:
        return None
    return prediction


def split_roi_prediction(self, prediction, splits):
    chunks = {}
    for k, v in prediction.items():
        res = _base_split(self, v, splits)
        chunks[k] = res

    return [{k: v[i] for k, v in chunks.items()} for i in range(len(splits))]


def get_single_roi_loss(self, stage, pred, img_metas, teacher=None):
    loss = dict()
    if self.with_bbox:
        bbox_results = pred["bbox_results"]
        bbox_targets = pred["bbox_targets"]
        rois = pred["rois"]
        if teacher is not None:
            unsup_flag = ["unsup" in meta["tag"] for meta in img_metas]
            if sum(unsup_flag) > 0:
                with torch.no_grad():
                    label_weights = bbox_targets[1].detach().clone()
                    proposal_list = roi2bbox(rois)
                    label_weights = torch.split(
                        label_weights, [len(p) for p in proposal_list], dim=0
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
                    if (
                        teacher.train_cfg.get("rate_position", "per_stage")
                        == "per_stage"
                    ):
                        rated_weights = teacher.rate(**unsup_proposals, stage=stage)
                    else:
                        rated_weights = teacher.rate(**unsup_proposals)
                    unsup_inds = 0
                    for i, flag in enumerate(unsup_flag):
                        if flag:
                            if (
                                teacher.train_cfg.get("rate_method", "background")
                                == "background"
                            ):
                                neg_inds = (
                                    label_list[i] == self.bbox_head[stage].num_classes
                                )
                                label_weights[i][neg_inds] = (
                                    label_weights[i][neg_inds]
                                    * rated_weights[unsup_inds][:, -1][neg_inds]
                                )
                            elif (
                                self.train_cfg.get("rate_method", "background")
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
            self.bbox_head[stage].loss(
                bbox_results["cls_score"],
                bbox_results["bbox_pred"],
                rois,
                *bbox_targets,
            )
        )
    return loss


def get_roi_loss(self, pred, img_metas, teacher=None):
    loss = {}
    for i in range(self.num_stages):
        lw = self.stage_loss_weights[i]
        cur_loss = get_single_roi_loss(self, i, pred[f"s{i}"], img_metas, teacher)
        for name, l in cur_loss.items():
            loss[f"s{i}.{name}"] = l * lw if "loss" in name else l
    return loss
