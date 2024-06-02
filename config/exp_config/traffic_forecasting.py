exp_conf = dict(
    hist_len=12,
    pred_len=12,

    batch_size=64,
    max_epoch=100,
    lr=0.002,
    optimizer="Adam",
    optimizer_weight_decay=0.0001,
    lr_scheduler='ReduceLROnPlateau',
    lrs_factor=0.5,
    lrs_patience=3,
    gradient_clip_val=5,
    val_metric="val/loss",
    null_value=0.0,
    test_metric="test/mae",
    test_timestep=[3, 6, 12],
    es_patience=10,
    use_norm_time_marker=True,

    num_workers=2,
    save_root="save",
    data_root="dataset",
)
