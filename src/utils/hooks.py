import wandb
from math import cos, pi
import mmcv
from mmcv.parallel import is_module_wrapper
from mmcv.runner import get_dist_info, master_only
from mmcv.runner.hooks import (
    HOOKS,
    Hook,
    OptimizerHook,
    WandbLoggerHook,
)

try:
    import apex
except:
    print("apex is not installed")


@HOOKS.register_module()
class DistDaliSamplerSeedHook(Hook):
    def before_epoch(self, runner):
        if hasattr(runner.data_loader, "reset"):
            runner.data_loader.reset(runner.epoch)
        else:
            if hasattr(runner.data_loader.sampler, "set_epoch"):
                # in case the data loader uses `SequentialSampler` in Pytorch
                runner.data_loader.sampler.set_epoch(runner.epoch)
            elif hasattr(runner.data_loader.batch_sampler.sampler, "set_epoch"):
                # batch sampler in pytorch warps the sampler as its attributes.
                runner.data_loader.batch_sampler.sampler.set_epoch(runner.epoch)


@HOOKS.register_module()
class ApexFP16OptimizerHook(OptimizerHook):
    """Optimizer hook for distributed training."""

    def __init__(self, update_interval=1, grad_clip=None, **kwargs):
        self.grad_clip = grad_clip
        self.update_interval = update_interval

    def before_run(self, runner):
        for m in runner.model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True
        runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        runner.outputs["loss"] /= self.update_interval
        with apex.amp.scale_loss(
            runner.outputs["loss"], runner.optimizer
        ) as scaled_loss:
            scaled_loss.backward()
        if self.every_n_iters(runner, self.update_interval):
            if self.grad_clip is not None:
                self.clip_grads(runner.model.parameters())
            runner.optimizer.step()
            runner.optimizer.zero_grad()


@HOOKS.register_module()
class MeanTeacherHook(Hook):
    """Hook for Mean Teacher
    This hook includes momentum adjustment following:
        m = 1 - ( 1- m_0) * (cos(pi * k / K) + 1) / 2,
        k: current step, K: total steps.
    Args:
        end_momentum (float): The final momentum coefficient
            for the target network. Default: 1.
    """

    def __init__(
        self,
        end_momentum=1.0,
        update_interval=1,
        warmup=True,
        warmup_step=100,
        dynamic=False,
        **kwargs,
    ):

        self.end_momentum = end_momentum
        self.update_interval = update_interval
        self.dynamic = dynamic
        self.warmup = warmup and (warmup_step > 0)
        self.warmup_step = warmup_step

    def before_train_iter(self, runner):
        assert hasattr(
            runner.model.module, "momentum"
        ), 'The runner must have attribute "momentum" in Model.'
        assert hasattr(
            runner.model.module, "base_momentum"
        ), 'The runner must have attribute "base_momentum" in Model.'
        if self.every_n_iters(runner, self.update_interval) and self.warmup:
            cur_iter = runner.iter
            base_m = runner.model.module.base_momentum
            if cur_iter < self.warmup_step:
                m = base_m * 1.0 * cur_iter / self.warmup_step
            else:
                if self.dynamic:
                    max_iter = runner.max_iters
                    m = (
                        self.end_momentum
                        - (self.end_momentum - base_m)
                        * (cos(pi * cur_iter / float(max_iter)) + 1)
                        / 2
                    )
                else:
                    m = base_m
            runner.model.module.momentum = m
        if self.every_n_iters(runner, self.update_interval):
            runner.log_buffer.update(dict(weight_momentum=runner.model.module.momentum))
            if is_module_wrapper(runner.model):
                runner.model.module.momentum_update()
            else:
                runner.model.momentum_update()


@HOOKS.register_module()
class GlobalWandbLoggerHook(WandbLoggerHook):
    buffer = dict()
    _history = dict()
    _mem = dict()
    _iter = -1
    _observe_iter = -1
    _interval = 1
    _initialized = False

    def __init__(
        self,
        init_kwargs=None,
        interval=10,
        ignore_last=True,
        reset_flag=True,
        commit=True,
        by_epoch=True,
    ):
        super(GlobalWandbLoggerHook, self).__init__(
            init_kwargs, interval, ignore_last, reset_flag, commit, by_epoch
        )
        GlobalWandbLoggerHook._initialized = True
        GlobalWandbLoggerHook._interval = 10 * interval

    @master_only
    def before_run(self, runner):
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            init_kwargs = self.init_kwargs
        else:
            init_kwargs = dict()
        is_watching = init_kwargs.pop("is_watching", False)

        if len(GlobalWandbLoggerHook.buffer) > 0:
            if "config" in GlobalWandbLoggerHook.buffer:
                cfg = GlobalWandbLoggerHook.buffer["config"]
                if hasattr(cfg, "to_dict"):
                    cfg = cfg.to_dict()
                init_kwargs.update(config=cfg)
            if "name" in GlobalWandbLoggerHook.buffer:
                init_kwargs.update(name=GlobalWandbLoggerHook.buffer["name"])

        self.wandb.init(**self.init_kwargs)

        if is_watching:
            self.wandb.watch(runner.model, log_freq=1)

    def before_iter(self, runner):
        super(GlobalWandbLoggerHook, self).before_iter(runner)
        GlobalWandbLoggerHook._iter = runner.iter

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            if len(GlobalWandbLoggerHook._history) > 0:
                self.wandb.log(
                    GlobalWandbLoggerHook._history, step=self.get_iter(runner)
                )
                GlobalWandbLoggerHook.reset()
            self.wandb.log(tags, step=self.get_iter(runner), commit=self.commit)
            # log gradient

    @classmethod
    def convert_box(cls, tag, boxes, box_labels, class_labels, std, scores=None):
        if isinstance(std, int):
            std = [std, std]
        if len(std) != 4:
            std = std[::-1] * 2
        std = boxes.new_tensor(std).reshape(1, 4)
        wandb_box = {}
        boxes = boxes / std
        boxes = boxes.detach().cpu().numpy().tolist()
        box_labels = box_labels.detach().cpu().numpy().tolist()
        class_labels = {k: class_labels[k] for k in range(len(class_labels))}
        wandb_box["class_labels"] = class_labels
        assert len(boxes) == len(box_labels)
        if scores is not None:
            scores = scores.detach().cpu().numpy().tolist()
            box_data = [
                dict(
                    position=dict(minX=box[0], minY=box[1], maxX=box[2], maxY=box[3]),
                    class_id=label,
                    scores=dict(cls=scores[i]),
                )
                for i, (box, label) in enumerate(zip(boxes, box_labels))
            ]
        else:
            box_data = [
                dict(
                    position=dict(minX=box[0], minY=box[1], maxX=box[2], maxY=box[3]),
                    class_id=label,
                )
                for i, (box, label) in enumerate(zip(boxes, box_labels))
            ]

        wandb_box["box_data"] = box_data
        return {tag: wandb.data_types.BoundingBoxes2D(wandb_box, tag)}

    @classmethod
    def convert_mask(
        cls, tag, masks, mask_labels, class_labels,
    ):
        label = {k: l for k, l in enumerate(("bg",) + class_labels)}
        vis_mask = {
            f"{tag}_{i}": {
                "mask_data": masks[i] * (1 + mask_labels[i].detach().cpu().numpy()),
                "class_labels": label,
            }
            for i in range(len(masks))
        }
        return vis_mask

    @classmethod
    def add_image(cls, key, image, img_norm_cfg=None):
        def color_transform(img_tensor, mean, std, to_rgb=False):
            img_np = img_tensor.detach().cpu().numpy().transpose((1, 2, 0))
            return mmcv.imdenormalize(img_np, mean, std, to_bgr=not to_rgb)

        if not isinstance(image, list):
            image = [image]
        if img_norm_cfg is not None:
            for im in image:
                im["data_or_path"] = color_transform(im["data_or_path"], **img_norm_cfg)
        GlobalWandbLoggerHook._history[key] = [wandb.Image(**im) for im in image]

    @classmethod
    def add_scalars(cls, scalar_dict):
        GlobalWandbLoggerHook._history.update(scalar_dict)

    @classmethod
    def accumulate_scalars(cls, scalar_dict):
        for k, v in scalar_dict.items():
            if k not in GlobalWandbLoggerHook._mem:
                GlobalWandbLoggerHook._mem[k] = v
            else:
                GlobalWandbLoggerHook._mem[k] += v

    @classmethod
    def access_mem(cls, key=None):
        if key is None:
            return GlobalWandbLoggerHook._mem
        else:
            return GlobalWandbLoggerHook._mem[key]

    @classmethod
    def is_display_time(cls):
        return (
            GlobalWandbLoggerHook._initialized
            and (
                (GlobalWandbLoggerHook._iter + 1) % GlobalWandbLoggerHook._interval == 0
            )
            and (get_dist_info()[0] == 0)
        )

    @classmethod
    def flush_global_buffer(cls, key, value):
        @master_only
        def _update():
            GlobalWandbLoggerHook.buffer[key] = value

        _update()

    @classmethod
    def reset(cls):
        GlobalWandbLoggerHook.buffer = dict()
        GlobalWandbLoggerHook._history = dict()
        GlobalWandbLoggerHook._observe_iter = GlobalWandbLoggerHook._iter

    @classmethod
    def get_global_buffer(cls, key):
        @master_only
        def _get():
            return GlobalWandbLoggerHook.buffer[key]

        return _get()

    @classmethod
    def iteration(cls):
        return GlobalWandbLoggerHook._iter
