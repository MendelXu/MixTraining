_base_ = [
    "../_base_/datasets/coco_detection_rgb.py",
    "../_base_/models/faster_rcnn_swin_small.py",
    "../_base_/runtimes/default_runtime.py",
    "../_base_/schedules/swin_adamw_schedule_1x.py",
]

data = dict(samples_per_gpu=2, workers_per_gpu=2,)

lr_config = dict(step=[64, 88])
runner = dict(max_epochs=96)

custom_hooks = [dict(type="NumClassCheckHook")]
