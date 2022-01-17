_base_ = [
    "../_base_/datasets/coco_detection_default.py",
    "https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.11.0/configs/_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/runtimes/default_runtime.py",
    "../_base_/schedules/sgd_schedule_1x.py",
]
model = dict(
    pretrained="open-mmlab://detectron2/resnet50_caffe",
    backbone=dict(norm_cfg=dict(requires_grad=False), norm_eval=True, style="caffe"),
)
data = dict(samples_per_gpu=2, workers_per_gpu=2,)

lr_config = dict(step=[16 * 10, 22 * 10])
runner = dict(max_epochs=24 * 20)
custom_hooks = [dict(type="NumClassCheckHook")]
resume_from = "work_dirs/supervised_v2/faster_rcnn_r50_caffe_fpn_coco_4x/epoch_128.pth"
