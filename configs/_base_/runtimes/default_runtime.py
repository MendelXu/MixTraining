checkpoint_config = dict(interval=1, create_symlink=False)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="GlobalWandbLoggerHook",
            init_kwargs=dict(
                project="self_distil",
                entity="sdl",
                name="{exp_name}",
                # log some important params
                config=dict(work_dir="{work_dir}"),
                notes="{note}",
            ),
        ),
    ],
)
# yapf:enable
# fp16
fp16 = dict(loss_scale="dynamic")
# momentum update
custom_hooks = [dict(type="NumClassCheckHook"), dict(type="MeanTeacherHook")]
# custom
dist_params = dict(backend="nccl")
log_level = "INFO"
auto_resume = True
load_from = None
resume_from = None
workflow = [("train", 1)]
note = ""
