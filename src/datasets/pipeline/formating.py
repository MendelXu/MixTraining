import torch
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.formating import Collect
import numpy as np


@PIPELINES.register_module()
class ExtraAttrs(object):
    def __init__(self, **attrs):
        self.attrs = attrs

    def __call__(self, results):
        for k, v in self.attrs.items():
            assert k not in results
            results[k] = v
        return results


@PIPELINES.register_module()
class PlainCollect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
    """

    def __init__(
        self,
        keys=[
            "img",
            "gt_bboxes",
            "gt_labels",
            "filename",
            "trans_matrix",
            "ori_filename",
            "flip",
            "flip_direction",
            "img_shape",
            "scale_factor",
        ],
        extra_keys=[],
    ):
        self.keys = keys + extra_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys}"


@PIPELINES.register_module()
class CollectV1(Collect):
    def __init__(self, *args, extra_meta_keys=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_keys = self.meta_keys + tuple(extra_meta_keys)


@PIPELINES.register_module()
class PseudoSamples(object):
    def __init__(self, with_bbox=False, with_mask=False, with_seg=False, override=True):
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.override = override

    def __call__(self, results):
        if self.with_bbox:
            if self.override and "gt_bboxes" in results:
                results.pop("gt_bboxes")
            if "gt_bboxes" not in results:
                results["gt_bboxes"] = np.zeros((0, 4))
                results["gt_labels"] = np.zeros((0,))
            if "bbox_fields" not in results:
                results["bbox_fields"] = []
            if "gt_bboxes" not in results["bbox_fields"]:
                results["bbox_fields"].append("gt_bboxes")
        if self.with_mask:
            if self.override and "gt_masks" in results:
                results.pop("gt_masks")
            if "gt_masks" not in results:
                # TODO: keep consistent with original pipeline, use Bitmasks
                results["gt_masks"] = np.zeros((0, 1, 1))

            if "mask_fields" not in results:
                results["mask_fields"] = []
            if "gt_masks" not in results["mask_fields"]:
                results["mask_fields"].append("gt_masks")
        if self.with_seg:
            if self.override and "gt_semantic_seg" in results:
                results.pop("gt_semantic_seg")
            if "gt_semantic_seg" not in results:
                results["gt_semantic_seg"] = 255 * np.ones(
                    results["img"].shape[:2], dtype=np.uint8
                )
            if "seg_fields" not in results:
                results["seg_fields"] = []
            if "gt_semantic_seg" not in results["seg_fields"]:
                results["seg_fields"].append("gt_semantic_seg")
        return results
