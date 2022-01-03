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


color_transform = dict(
    type="RandomApply",
    policies=[
        dict(type="Identity"),
        dict(type="Jitter", contrast=1.0),
        dict(type="Jitter", brightness=1.0),
        dict(type="Jitter", hue=1.0),
        dict(type="Equalize"),
        dict(type="AutoContrast"),
        dict(type="PosterizeV1"),
        dict(type="RandomGrayScale"),
        dict(type="SolarizeV1"),
    ],
)

geo_transform = dict(
    type="RandomApply",
    policies=[
        dict(type="Identity"),
        dict(
            type="ImTranslate",
            level=5,
            prob=1.0,
            max_translate_offset=100,
            direction="horizontal",
        ),
        dict(
            type="ImTranslate",
            level=5,
            prob=1.0,
            max_translate_offset=100,
            direction="vertical",
        ),
        dict(type="ImRotate", level=5, prob=1.0),
        dict(type="ImShear", level=5, prob=1.0),
    ],
)

img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
scale_cfg = dict(
    img_scale=[(1333, 400), (1333, 1200)], multiscale_mode="range", keep_ratio=True
)
data_root = "data/coco"
sup_set = "train2017"
val_set = "val2017"
test_set = "val2017"
# end def

sup = [
    dict(type="LoadImageFromFile", file_client_args=dict(backend="zip")),
    dict(type="LoadAnnotations", with_bbox=True),
    color_transform,
    dict(type="Resize", **scale_cfg),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    geo_transform,
    dict(
        type="CutOut",
        n_holes=(1, 5),
        cutout_ratio=[
            (0.05, 0.05),
            (0.75, 0.75),
            (0.1, 0.1),
            (0.125, 0.125),
            (0.15, 0.15),
            (0.175, 0.175),
            (0.2, 0.2),
        ],
        fill_in=(0, 0, 0),
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="ExtraAttrs", tag="sup"),
    dict(
        type="CollectV1",
        keys=["img", "gt_bboxes", "gt_labels"],
        extra_meta_keys=["tag"],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="CocoDataset",
        ann_file="{data_root}/annotations/instances_{sup_set}.json",
        img_prefix="{data_root}/{sup_set}/",
        pipeline=sup,
    ),
)

lr_config = dict(step=[8, 11])
runner = dict(max_epochs=12)

custom_hooks = [dict(type="NumClassCheckHook")]
