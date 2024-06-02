import os

import numpy as np
from torch.utils.data import Dataset


class GeneralTSFDataset(Dataset):
    """
    General TSF Dataset.
    """

    def __init__(self, data_root, dataset_name, hist_len, pred_len, data_split, freq, mode, use_norm_time_marker=True):
        self.data_dir = os.path.join(data_root, dataset_name)
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.train_len, self.val_len, self.test_len = data_split
        self.freq = freq

        self.mode = mode
        assert mode in ['train', 'valid', 'test'], "mode {} mismatch, should be in [train, valid, test]".format(mode)

        mode_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = mode_map[mode]

        self.use_norm_time_marker = use_norm_time_marker

        self.features = self.__read_data__()

    def __read_data__(self):
        norm_feature_path = os.path.join(self.data_dir, 'feature.npz')
        norm_feature = np.load(norm_feature_path)

        norm_var = norm_feature['norm_var']
        if self.use_norm_time_marker:
            time_marker = norm_feature['norm_time_marker']
        else:
            time_marker = norm_feature['time_marker']

        border1s = [0, self.train_len, self.train_len + self.val_len]
        border2s = [self.train_len, self.train_len + self.val_len, norm_var.shape[0]]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        norm_var = norm_var[border1:border2]
        time_marker = time_marker[border1:border2]

        L, N = norm_var.shape
        norm_var = norm_var[:, :, np.newaxis]
        time_marker = time_marker[:, np.newaxis, :].repeat(N, axis=1)

        features = np.concatenate([norm_var, time_marker], axis=-1)  # (L, N, C), C=4, [speed, tod, dow, dom, doy]
        return features

    def __getitem__(self, index):
        hist_start = index
        hist_end = index + self.hist_len
        pred_end = hist_end + self.pred_len
        return self.features[hist_start:hist_end, ...], self.features[hist_end:pred_end, ...]

    def __len__(self):
        return len(self.features) - (self.hist_len + self.pred_len) + 1


def data_provider(config, mode):
    return GeneralTSFDataset(
        data_root=config['data_root'],
        dataset_name=config['dataset_name'],
        hist_len=config['hist_len'],
        pred_len=config['pred_len'],
        data_split=config['data_split'],
        freq=config['freq'],
        mode=mode,
        use_norm_time_marker=config['use_norm_time_marker']
    )
