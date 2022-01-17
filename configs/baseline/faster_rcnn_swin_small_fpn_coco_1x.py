_base_ = [
    "../_base_/datasets/coco_detection_rgb.py",
    "../_base_/models/faster_rcnn_swin_small.py",
    "../_base_/runtimes/default_runtime.py",
    "../_base_/schedules/swin_adamw_schedule_1x.py",
]

data = dict(samples_per_gpu=2, workers_per_gpu=2,)

lr_config = dict(step=[16, 22])
runner = dict(max_epochs=24)

custom_hooks = [dict(type="NumClassCheckHook")]

resume_from = "work_dirs/supervised/faster_rcnn_swin_small_fpn_coco_4x/epoch_16.pth"
