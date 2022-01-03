import torch
from torch import nn
from torch.nn import functional as F
from mmdet.core import multi_apply
from src.core import Transform2D, filter_invalid
from src.utils.structure_utils import pad_stack


class Teacher(nn.Module):
    def __init__(self, detector, requires_grad=False, eval=True):
        super().__init__()
        self._fields = []
        self.detector = detector
        # freeze detector
        if not requires_grad:
            for param in self.detector.parameters():
                param.requires_grad = False
        # deal with detector with dropout and bn
        if eval:
            self.detector.eval()

    @torch.no_grad()
    def read(self, img, **kwargs):
        self.img_metas = kwargs.get("img_metas")
        feat = self.detector.extract_feats(img)
        # store data in own
        self.backbone_feat = feat

    @torch.no_grad()
    def learn(self, student, momentum=None):
        if momentum is not None:
            for src_parm, tgt_parm in zip(
                student.detector.parameters(), self.detector.parameters()
            ):
                ori_type = tgt_parm.data.dtype
                tgt_parm.data = (
                    tgt_parm.data.float() * momentum
                    + src_parm.data.float() * (1.0 - momentum)
                ).to(ori_type)

    @torch.no_grad()
    def deliver(self, student_input, student_infos):
        # connect student to each teacher
        pseudo_gt = {k: [] for k in self._fields}
        for sinfo in student_infos:
            student_id = sinfo["ori_filename"]
            teacher_idx = self.query_teacher_by_id(student_id)
            trans_matrix = [
                torch.matmul(
                    torch.from_numpy(sinfo["trans_matrix"]).to(
                        self.backbone_feat[0].device
                    ),
                    torch.from_numpy(self.img_metas[idx]["trans_matrix"])
                    .to(self.backbone_feat[0].device)
                    .inverse(),
                )
                for idx in teacher_idx
            ]
            output_shape = [sinfo["img_shape"][:2] for _ in teacher_idx]
            for name in self._fields:
                res = [getattr(self, name)[idx] for idx in teacher_idx]
                if any([r is None for r in res]):
                    continue
                if name == "gt_bboxes":
                    res = Transform2D.transform_bboxes(
                        [r.detach() for r in res], trans_matrix, output_shape
                    )
                elif name == "gt_masks":
                    res = Transform2D.transform_masks(res, trans_matrix, output_shape)
                elif name == "gt_semantic_seg":
                    res = Transform2D.transform_image(
                        [r.detach() + 1 for r in res], trans_matrix, output_shape
                    )
                    res = [r - 1 for r in res]

                else:
                    res = [r.detach() for r in res]

                if len(res) > 1:
                    raise NotImplementedError()
                    # res = fuse(res)
                else:
                    res = res[0]
                if name == "gt_semantic_seg":
                    res[res == -1] = 255
                pseudo_gt[name].append(res)

        (
            pseudo_gt["gt_bboxes"],
            pseudo_gt["gt_labels"],
            pseudo_gt["gt_masks"],
        ) = multi_apply(
            filter_invalid,
            pseudo_gt["gt_bboxes"],
            pseudo_gt["gt_labels"],
            [None for _ in student_infos],
            pseudo_gt.get("gt_masks", [None for _ in student_infos]),
        )
        for key in list(pseudo_gt.keys()):
            if key not in self._fields:
                pseudo_gt.pop(key)
        if "gt_semantic_seg" in pseudo_gt:
            pseudo_gt["gt_semantic_seg"] = pad_stack(
                pseudo_gt["gt_semantic_seg"], student_input.shape[-2:]
            )
            pseudo_gt["gt_semantic_seg"] = F.interpolate(
                pseudo_gt["gt_semantic_seg"].unsqueeze(1).float(), scale_factor=0.125
            ).long()
        if "gt_masks" in pseudo_gt:
            pseudo_gt["gt_masks"] = [
                mask.pad(student_info["pad_shape"][:2])
                for mask, student_info in zip(pseudo_gt["gt_masks"], student_infos)
            ]
        return pseudo_gt

    def rate(self):
        pass

    def query_teacher_by_id(self, file_id):
        teacher_ids = [meta["ori_filename"] for meta in self.img_metas]
        return [teacher_ids.index(file_id)]


class Student(nn.Module):
    def __init__(self, detector, train_cfg=None):
        super().__init__()
        self.detector = detector
        self.train_cfg = train_cfg

    def learn(self, problem_data, teacher):
        pass

    def self_learn(self, reference_data):
        pass

    def parallel_learn(self, labeled, unlabeled, teacher):
        pass
