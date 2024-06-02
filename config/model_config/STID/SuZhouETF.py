model_conf = dict(
    model_name="STID",
    dataset_name="SuZhouETF",
    exp_type="traffic_forecasting",

    block_num=3,
    ts_emb_dim=32,
    node_emb_dim=32,
    tod_emb_dim=32,
    dow_emb_dim=32,

    lr=0.002
)