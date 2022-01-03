import torch
from mmdet.core import multi_apply
from mmdet.models.roi_heads import CascadeRoIHead, StandardRoIHead
from . import standard_wrappers, cascade_wrappers


def simple_test_bboxes(self, x, img_metas, proposal_list, **kwargs):
    if isinstance(self, CascadeRoIHead):
        return cascade_wrappers.get_bbox_confidence(
            self, x, img_metas, proposal_list, **kwargs
        )
    elif isinstance(self, StandardRoIHead):
        return standard_wrappers.get_bbox_confidence(
            self, x, img_metas, proposal_list, **kwargs
        )
    else:
        raise NotImplementedError(
            f"confidence estimation method for {type(self)} is not implemented yet."
        )


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
    if isinstance(self, CascadeRoIHead):
        return cascade_wrappers.get_roi_prediction(
            self,
            x,
            proposal_list,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_masks,
            gt_bboxes_ignore,
            **kwargs,
        )
    elif isinstance(self, StandardRoIHead):
        return standard_wrappers.get_roi_prediction(
            self,
            x,
            proposal_list,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_masks,
            gt_bboxes_ignore,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            f"get_prediction method for {type(self)} is not implemented yet."
        )


def get_roi_loss(self, pred, img_metas, **kwargs):
    if isinstance(self, CascadeRoIHead):
        return cascade_wrappers.get_roi_loss(self, pred, img_metas, **kwargs)
    elif isinstance(self, StandardRoIHead):
        return standard_wrappers.get_roi_loss(self, pred, img_metas, **kwargs)
    else:
        raise NotImplementedError()


def _split(*input, splits=None):
    if splits is None:
        return input
    else:
        return tuple([torch.split(i, splits) for i in input])


def split_rpn_output(rpn_output, splits):
    if rpn_output is None:
        return [None for _ in splits]
    else:
        tmp = multi_apply(_split, *rpn_output, splits=splits)
        return [[[tt[j] for tt in t] for t in tmp] for j in range(len(splits))]


def split_roi_prediction(self, roi_output, splits):
    if isinstance(self, CascadeRoIHead):
        return cascade_wrappers.split_roi_prediction(self, roi_output, splits)
    elif isinstance(self, StandardRoIHead):
        return standard_wrappers.split_roi_prediction(self, roi_output, splits)
    else:
        raise NotImplementedError(
            f"confidence estimation method for {type(self)} is not implemented yet."
        )
