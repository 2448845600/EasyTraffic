ShenZhenETF_conf = dict(
    dataset_name='ShenZhenETF',
    var_num=41,
    freq=10,
    data_split=[26064, 13248, 13104],  # (31+28+31+30+31+30)*24*6, (31+31+30)*24*6, (31+30+31-1)*24*6
    description="The ShenZhen2000 raw_dataset is urban road speed data. This raw_dataset contains road speed data within 2,000 meters around the Shenzhen Convention and Exhibition Center in Futian District, Shenzhen City in 2018, collected every 10 minutes.",
)

SuZhouETF_conf = dict(
    dataset_name='SuZhouETF',
    var_num=8,
    freq=10,
    data_split=[26064, 13248, 13104],  # (31+28+31+30+31+30)*24*6, (31+31+30)*24*6, (31+30+31-1)*24*6
    description="The SuZhou2000 raw_dataset is urban road speed data. This raw_dataset contains road speed data within 2,000 meters around the SuZhou International Expo Center, SuZhou City in 2018, collected every 10 minutes.",
)

PEMSBAY_conf = dict(
    dataset_name='PEMSBAY',
    var_num=325,
    freq=5,
    data_split=[36481, 5212, 10423],
)

METRLA_conf = dict(
    dataset_name='METRLA',
    var_num=207,
    freq=5,
    data_split=[23990, 3428, 6854],
)

PEMS04_conf = dict(
    dataset_name='PEMS04',
    var_num=307,
    freq=5,
    data_split=[10195, 3399, 3398],
)

ETTh1_conf = dict(
    dataset_name='ETTh1',
    var_num=7,
    freq=60,
    data_split=[8640, 2880, 2880],
    description="The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment. This dataset consists of 2 years data from two separated counties in China. To explore the granularity on the Long sequence time-series forecasting (LSTF) problem, different subsets are created, {ETTh1, ETTh2} for 1-hour-level and ETTm1 for 15-minutes-level. Each data point consists of the target value ”oil temperature” and 6 power load features. The train/val/test is 12/4/4 months."
)

ETTh2_conf = dict(
    dataset_name='ETTh2',
    var_num=7,
    freq=60,
    data_split=[8640, 2880, 2880],
)

ETTm1_conf = dict(
    dataset_name='ETTm1',
    var_num=7,
    freq=15,
    data_split=[34560, 11520, 11520],
)


