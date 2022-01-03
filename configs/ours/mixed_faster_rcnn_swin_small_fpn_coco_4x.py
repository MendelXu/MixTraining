_base_ = [
    "../_base_/datasets/disl_coco_detection_rgb.py",
    "../_base_/models/mixed_faster_rcnn_swin_small.py",
    "../_base_/runtimes/default_runtime.py",
    "../_base_/schedules/swin_adamw_schedule_1x.py",
]
lr_config = dict(step=[32, 44])
runner = dict(max_epochs=48)
data = dict(samples_per_gpu=2, workers_per_gpu=2)
