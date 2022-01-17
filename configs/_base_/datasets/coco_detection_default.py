# def temp
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
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    color_transform,
    dict(type="Resize", **scale_cfg),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="ExtraAttrs", tag="sup"),
    dict(
        type="CollectV1",
        keys=["img", "gt_bboxes", "gt_labels"],
        extra_meta_keys=["tag"],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="CocoDataset",
        ann_file="{data_root}/annotations/instances_{sup_set}.json",
        img_prefix="{data_root}/{sup_set}/",
        pipeline=sup,
    ),
    val=dict(
        type="CocoDataset",
        ann_file="{data_root}/annotations/instances_{val_set}.json",
        img_prefix="{data_root}/{val_set}",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CocoDataset",
        ann_file="{data_root}/annotations/instances_{test_set}.json",
        img_prefix="{data_root}/{test_set}",
        pipeline=test_pipeline,
    ),
)

evaluation = dict(gpu_collect=True, metric=["bbox"])
