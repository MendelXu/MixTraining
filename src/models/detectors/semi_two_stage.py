import torch
from mmdet.models import DETECTORS, build_detector
from mmdet.models.detectors import BaseDetector
from src.utils import GlobalWandbLoggerHook
from src.utils.debug_utils import Timer
from src.utils.structure_utils import check_equal, dict_concat, dict_split, zero_like

from .student_wrapper import TwoStageStudent
from .teacher_wrapper import TwoStageTeacher


@DETECTORS.register_module()
class SemiTwoStageDetector(BaseDetector):
    def __init__(
        self,
        student_cfg,
        teacher_cfg=None,
        train_cfg=None,
        test_cfg=None,
        base_momentum=0.999,
    ):
        super().__init__()
        if teacher_cfg is None:
            teacher_cfg = student_cfg
        teacher_detector = build_detector(teacher_cfg)
        self.teacher = TwoStageTeacher(teacher_detector, train_cfg=train_cfg)
        student_detector = build_detector(student_cfg)
        self.student = TwoStageStudent(student_detector, train_cfg=train_cfg)
        # self.student.register_teacher_supervision(self.teacher)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg is None:
            self.train_cfg = {}
        if self.test_cfg is None:
            self.test_cfg = {}

        self.base_momentum = base_momentum
        self.momentum = self.base_momentum
        if self.base_momentum < 1:
            check_equal(self.teacher.detector, self.student.detector)
            self._momentum_update(0.0)

    @torch.no_grad()
    def _momentum_update(self, momentum):
        """Momentum update of the target network."""
        self.teacher.learn(self.student, momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update(self.momentum)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        return self.student.detector.forward_dummy(img)

    def forward_train(self, img, img_metas, **kwargs):

        with Timer("split data"):
            if not hasattr(self.teacher, "CLASSES"):
                self.teacher.CLASSES = self.CLASSES
            if not hasattr(self.student, "CLASSES"):
                self.student.CLASSES = self.CLASSES
            unsup_tag = self.train_cfg.get(
                "unsup_tag", ["unsup_teacher", "unsup_student"]
            )
            sup_tag = self.train_cfg.get("sup_tag", ["sup"])
            kwargs.update({"img": img})
            kwargs.update({"img_metas": img_metas})
            kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
            data_groups = dict_split(kwargs, "tag")
            sample_num = {}
            for tag in unsup_tag:
                sample_num[tag] = 0
            for tag in sup_tag:
                sample_num[tag] = 0

            for k, v in data_groups.items():
                sample_num[k] = len(v["img"])
            GlobalWandbLoggerHook.add_scalars(sample_num)

        if any([s in data_groups for s in sup_tag]):
            # compute supervised loss
            labeled_data_group = dict_concat(
                [data_groups[s] for s in sup_tag if s in data_groups]
            )
            labeled_data_group.pop("tag")

        else:
            labeled_data_group = None

        if unsup_tag[0] in data_groups:
            teacher_tag = unsup_tag[0]
            student_tag = unsup_tag[1]
            data_groups[teacher_tag].pop("tag")
            data_groups[student_tag].pop("tag")

            unlabeled_data = data_groups[student_tag]
        else:
            unlabeled_data = None
        with Timer("techer prediction"):
            if unlabeled_data is not None:
                self.teacher.read(data_groups[teacher_tag])
        loss = self.student.parallel_learn(
            labeled_data_group, unlabeled_data, self.teacher
        )
        return loss

    async def async_simple_test(self, img, img_metas, **kwargs):
        return self.inference_detector.async_simple_test(img, img_metas, **kwargs)

    def simple_test(self, img, img_metas, **kwargs):
        return self.inference_detector.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        return self.inference_detector.aug_test(imgs, img_metas, **kwargs)

    @property
    def inference_detector(self):
        if self.test_cfg.get("inference_on", "student") == "teacher":
            detector = self.teacher.detector
        else:
            detector = self.student.detector
        return detector

    def extract_feat(self, x):
        return self.student.detector.extract_feat(x)

    def extract_feats(self, imgs):
        return self.student.detector.extract_feats(imgs)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher.detector." + k: state_dict[k] for k in keys})
            state_dict.update({"student.detector." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
