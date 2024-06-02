import hashlib
import importlib


def get_model(config, scaler_list, adj_mx, meta_info):
    return getattr(importlib.import_module('easytf.model'), config['model_name'])(config, scaler_list, adj_mx, meta_info)


def cal_conf_hash(config, useless_key=None, hash_len=10):
    if useless_key is None:
        useless_key = ['save_root', 'data_root', 'seed', 'ckpt_path', 'conf_hash', 'use_wandb']

    conf_str = ''
    for k, v in config.items():
        if k not in useless_key:
            conf_str += str(v)

    md5 = hashlib.md5()
    md5.update(conf_str.encode('utf-8'))
    return md5.hexdigest()[:hash_len]
