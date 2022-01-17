_base_ = [
    "../_base_/datasets/coco_detection_default.py",
    "mmdet:configs/_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/runtimes/default_runtime.py",
    "../_base_/schedules/sgd_schedule_1x.py",
]
model = dict(
    pretrained="open-mmlab://detectron2/resnet50_caffe",
    backbone=dict(norm_cfg=dict(requires_grad=False), norm_eval=True, style="caffe"),
)
data = dict(samples_per_gpu=2, workers_per_gpu=2,)

lr_config = dict(step=[64 * 2, 88 * 2])
runner = dict(max_epochs=96 * 2)
custom_hooks = [dict(type="NumClassCheckHook")]
