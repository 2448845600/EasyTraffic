model_conf = dict(
    model_name="STID",
    dataset_name="PEMS04",
    exp_type="traffic_forecasting",

    block_num=3,
    ts_emb_dim=32,
    node_emb_dim=32,
    tod_emb_dim=32,
    dow_emb_dim=32,
    use_norm_time_marker=False,

    lr=0.002,
    batch_size=64,
)