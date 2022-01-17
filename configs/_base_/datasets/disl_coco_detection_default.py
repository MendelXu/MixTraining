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
geo_transform = dict(
    type="RandomApply",
    policies=[
        dict(type="LazyGeoIdentity"),
        dict(type="LazyTranslate", max_translate_offset=0.2, direction="horizontal",),
        dict(type="LazyTranslate", max_translate_offset=0.2, direction="veritical",),
        dict(type="LazyRotate"),
        dict(type="LazyShear"),
    ],
)
img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
scale_cfg = dict(
    img_scale=[(1333, 400), (1333, 1200)], multiscale_mode="range", keep_ratio=True
)
data_root = "data/coco"
sup_set = "train2017"
unsup_set = "train2017"
val_set = "val2017"
test_set = "test2017"
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
unsup_strong = [
    color_transform,
    dict(type="LazyResize", **scale_cfg),
    dict(type="LazyRandomFlip"),
    geo_transform,
    dict(type="Normalize", **img_norm_cfg),
    dict(type="TransformImage"),
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
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(
        type="CollectV1",
        keys=["img", "gt_bboxes", "gt_labels"],
        extra_meta_keys=["tag", "trans_matrix"],
    ),
]
unsup_weak = [
    dict(type="LazyResize", **scale_cfg),
    dict(type="LazyRandomFlip"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="TransformImage"),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(
        type="CollectV1",
        keys=["img", "gt_bboxes", "gt_labels"],
        extra_meta_keys=["tag", "trans_matrix"],
    ),
]
unsup = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="MultiBranch",
        policies=[
            dict(type="Compose", transforms=unsup_strong),
            dict(type="Compose", transforms=unsup_weak),
        ],
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
        type="MultiSourceDataset",
        datasets=[
            dict(
                type="CocoDataset",
                ann_file="{data_root}/annotations/instances_{sup_set}.json",
                img_prefix="{data_root}/{sup_set}/",
                pipeline=sup,
            ),
            dict(
                type="CocoDataset",
                ann_file="{data_root}/annotations/instances_{unsup_set}.json",
                img_prefix="{data_root}/{unsup_set}/",
                filter_empty_gt=False,
                pipeline=unsup,
            ),
        ],
        sample_ratio=[0.5, 0.5],
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
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            epoch_length=7330,
            by_prob=True,
            at_least_one=True,
        )
    ),
    loader=dict(train=None),
)

evaluation = dict(gpu_collect=True, metric=["bbox"])
