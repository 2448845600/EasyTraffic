import argparse
import importlib
import os

import lightning.pytorch as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from easytf.interface.base_data_interface import DataInterface
from easytf.util.util import cal_conf_hash

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(dataset_name, model_name, **kwargs):
    print('参数优先级: kwargs > model_conf > exp_conf = data_conf')
    print('exp_conf 与 data_conf 不能出现相同的参数')

    # 加载并更新 model_conf
    model_conf_module = importlib.import_module('config.model_config.{}.{}'.format(model_name, dataset_name))
    model_conf = model_conf_module.model_conf
    for k in kwargs:
        model_conf[k] = kwargs[k]

    # 加载 exp_conf
    exp_conf_module = importlib.import_module('config.exp_config.{}'.format(model_conf['exp_type']))
    exp_conf = exp_conf_module.exp_conf

    # 加载 data_conf
    data_conf_module = importlib.import_module('config.data_config')
    data_conf = eval('data_conf_module.{}_conf'.format(dataset_name))

    final_config = {**data_conf, **exp_conf}
    final_config.update(model_conf)
    final_config['conf_hash'] = cal_conf_hash(final_config, hash_len=10)
    return final_config


def load_callback(config):
    return [
        ModelCheckpoint(
            monitor=config.val_metric,
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
        ),
        EarlyStopping(
            monitor=config.val_metric,
            mode='min',
            patience=config.es_patience,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]


def train_func(config_dict):
    config = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(config, key, value)

    L.seed_everything(config.seed)

    save_dir = os.path.join(config.save_root, '{}_{}'.format(config.model_name, config.dataset_name))
    if config.use_wandb:
        run_logger = WandbLogger(save_dir=save_dir, name=config.conf_hash, version='seed_{}'.format(config.seed))
    else:
        run_logger = CSVLogger(save_dir=save_dir, name=config.conf_hash, version='seed_{}'.format(config.seed))

    trainer = L.Trainer(
        default_root_dir=config.save_root,
        accelerator="auto",
        precision="bf16-mixed" if config.model_name == "TimeLLM" else "32-true",
        devices=config.devices,
        max_epochs=config.max_epoch,
        callbacks=load_callback(config),
        logger=run_logger,
        gradient_clip_algorithm="Norm",
        gradient_clip_val=config.gradient_clip_val,
    )

    if config.exp_type == 'long_time_series_forecasting':
        from easytf.interface.long_time_series_forecasting_interface import ExpInterface
    elif config.exp_type == 'traffic_forecasting':
        from easytf.interface.traffic_forecasting_interface import ExpInterface
    else:
        raise "ModelInterface Error"
    data_module = DataInterface(**vars(config))
    model = ExpInterface(**vars(config))
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", default='ETTh1', type=str)
    parser.add_argument("-m", "--model_name", default='STLLM', type=str)
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--data_root", default="dataset", type=str, help="data root")
    parser.add_argument("--save_root", default="save", help="save root")
    parser.add_argument("--use_wandb", default=0, type=int, help="use wandb")
    parser.add_argument("--devices", default='0,', type=str, help="The devices to use, detail rules is show in README")
    args = parser.parse_args()

    config = load_config(args.dataset_name, args.model_name, seed=args.seed, data_root=args.data_root,
                         save_root=args.save_root, devices=args.devices, use_wandb=args.use_wandb)

    train_func(config)
