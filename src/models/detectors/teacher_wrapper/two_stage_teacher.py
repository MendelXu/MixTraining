import torch
from mmdet.core import multi_apply
from mmdet.models.roi_heads import HybridTaskCascadeRoIHead
from src.core import Transform2D, filter_invalid, recover_mask
from src.models.utils.semi_wrapper import simple_test_bboxes
from src.utils.debug_utils import Timer
from src.utils.structure_utils import result2bbox, result2mask

from ..semi_base import Teacher


class TwoStageTeacher(Teacher):
    def __init__(self, detector, train_cfg):
        super().__init__(detector)
        self.train_cfg = train_cfg
        if self.train_cfg is not None:
            self._fields = self.train_cfg.get(
                "supervised_fields",
                ["gt_bboxes", "gt_labels", "gt_masks", "gt_semantic_seg"],
            )
            self.ignore_branch = []
            if "gt_bboxes" not in self._fields:
                self.ignore_branch.append("bbox")
            if "gt_labels" not in self._fields:
                self.ignore_branch.append("cls")
            self.ignore_branch.append("mask")
            self.ignore_branch.append("seg")
        else:
            self._fields = None

    @torch.no_grad()
    def read_feat(self, data):
        feat = self.detector.extract_feat(data["img"])
        self.backbone_feat = feat
        self.img_metas = data["img_metas"]

    @torch.no_grad()
    def read(self, data):
        with Timer("teacher rpn"):
            feat = self.detector.extract_feat(data["img"])
            # store data in own
            self.backbone_feat = feat
            if self.detector.with_rpn:
                proposal_list = self.detector.rpn_head.simple_test_rpn(
                    feat, data["img_metas"]
                )
            else:
                proposal_list = data["proposals"]

            self.img_metas = data["img_metas"]
            self.proposal_list = proposal_list
        # roi prediction
        # Note: There is a bug in the original cascade mask rcnn simple test function,
        # we should not flip the mask in the head. So here we have to create a fake meta infomation.
        # https://github.com/open-mmlab/mmdetection/issues/1466
        with Timer("teacher rcnn"):
            fake_meta = [
                {
                    "img_shape": meta["img_shape"],
                    "ori_shape": meta["ori_shape"],
                    "scale_factor": meta["scale_factor"],
                    "flip": False,
                    "flip_direction": "Bug",
                }
                for meta in data["img_metas"]
            ]
            results = self.detector.roi_head.simple_test(
                feat, proposal_list, fake_meta, rescale=False
            )
        with Timer("teacher filter bbox"):
            self._prepare_instance_label(results, data["img"].device)
        with Timer("teacher semantic"):
            if (
                "gt_semantic_seg" in self._fields
                and hasattr(self.detector.roi_head, "with_semantic")
                and self.detector.roi_head.with_semantic
            ):
                semantic_pred, semantic_feat = self.detector.roi_head.semantic_head(
                    self.backbone_feat
                )
                self._prepare_semantic_label(
                    semantic_pred,
                    data["img_metas"],
                    scale=data["img"].shape[-1] / semantic_pred.shape[-1],
                )
                self.semantic_feat = semantic_feat

    def _prepare_semantic_label(self, semantic_pred, img_metas, scale=1.0):
        score, label = semantic_pred.softmax(dim=1).max(dim=1)
        label[score < self.train_cfg.get("semantic_seg_thr", 0.9)] = 255
        self.gt_semantic_seg, _ = multi_apply(
            recover_mask, label, img_metas, scale=scale
        )

    def _prepare_instance_label(self, instance_pred, device):
        with Timer("result2instance"):
            if len(instance_pred[0]) == 2:
                bbox_result = [instance_pred[i][0] for i in range(len(instance_pred))]
                mask_result = [instance_pred[i][1] for i in range(len(instance_pred))]
            else:
                bbox_result = instance_pred
                mask_result = None
            bboxes, labels = multi_apply(result2bbox, bbox_result)
            if "gt_masks" in self._fields and (mask_result is not None):
                masks, _ = multi_apply(result2mask, mask_result)
            else:
                masks = [None for _ in range(len(bboxes))]
        with Timer("filter"):
            bboxes = [torch.from_numpy(bbox).to(device) for bbox in bboxes]
            labels = [torch.from_numpy(label).to(device) for label in labels]
            bboxes, labels, masks = multi_apply(
                filter_invalid,
                bboxes,
                labels,
                [bbox[:, 4] for bbox in bboxes],
                masks,
                thr=self.train_cfg.get("score_thr", 0.5),
            )

        self.gt_bboxes = bboxes
        self.gt_labels = labels
        self.gt_masks = masks

    @torch.no_grad()
    def rate(self, bboxes, img_metas=None, **kwargs):
        # mapping bboxes to teacher image space
        if img_metas is not None:
            results = []
            for i, sinfo in enumerate(img_metas):
                student_id = sinfo["ori_filename"]
                teacher_idx = self.query_teacher_by_id(student_id)[0]
                source2teacher = torch.from_numpy(
                    self.img_metas[teacher_idx]["trans_matrix"]
                ).to(self.backbone_feat[0].device)
                student2source = (
                    torch.from_numpy(sinfo["trans_matrix"])
                    .to(self.backbone_feat[0].device)
                    .inverse()
                )
                trans_matrix = torch.matmul(source2teacher, student2source)
                output_shape = sinfo["img_shape"][:2]
                res = Transform2D.transform_bboxes(
                    bboxes[i], trans_matrix, output_shape
                ).float()
                results.append(torch.cat([res, res.new_ones(res.shape[0], 1)], dim=1))
            bboxes = results
        if isinstance(self.detector.roi_head, HybridTaskCascadeRoIHead):
            kwargs.update({"semantic_feat": getattr(self, "semantic_feat", None)})
        score_pred = simple_test_bboxes(
            self.detector.roi_head, self.backbone_feat, self.img_metas, bboxes, **kwargs
        )

        return score_pred
