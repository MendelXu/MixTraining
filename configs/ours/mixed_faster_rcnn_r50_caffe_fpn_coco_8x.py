_base_ = [
    "../_base_/datasets/disl_coco_detection_default.py",
    "../_base_/models/mixed_faster_rcnn_r50_caffe_fpn.py",
    "../_base_/runtimes/default_runtime.py",
    "../_base_/schedules/sgd_schedule_1x.py",
]

lr_config = dict(step=[32 * 2, 44 * 2])
runner = dict(max_epochs=48 * 2)
