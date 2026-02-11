import torch      
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

def generate_seq2seq_from_tensor(data, x_offsets, y_offsets):
    """
    data: np.ndarray, shape (T, N, F_total)
    x_offsets: array of negative-to-zero indices (history)
    y_offsets: array of positive indices (future)

    Returns:
      x: (num_samples, input_len, N, F_total)
      y: (num_samples, output_len, N, 1)  # predict feature 0 (value)
    """
    T, N, F_total = data.shape
    x, y = [], []

    min_t = -x_offsets[0]           # history_len - 1
    max_t = T - y_offsets[-1]       # last t where we can take full future window

    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]          # (input_len, N, F_total)
        y_t_full = data[t + y_offsets, ...]     # (output_len, N, F_total)
        y_t = y_t_full[..., 0:1]                # assume feature 0 = 'value'
        x.append(x_t)
        y.append(y_t)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y
def add_time_features_daily(X, index, 
                            add_day_of_week=True,
                            add_day_of_month=True,
                            add_month=True):
    """
    X: (T, N, F0)  -- your base features (value + promos)
    index: DatetimeIndex of length T
    Returns:
        X_all: (T, N, F0 + time_features)
    """
    T, N, F0 = X.shape
    data_list = [X]

    # 1. Day of week  (one-hot: 7 dims)
    if add_day_of_week:
        dow = index.dayofweek.values  # 0..6
        dow_onehot = np.zeros((T, N, 7), dtype=np.float32)
        dow_onehot[np.arange(T), :, dow] = 1.0
        data_list.append(dow_onehot)

    # 2. Day of month (continuous 1–31 scaled to [0, 1])
    if add_day_of_month:
        dom = (index.day.values / 31.0).astype(np.float32)
        dom_feat = np.tile(dom[:, None, None], (1, N, 1))
        data_list.append(dom_feat)

    # 3. Month of year (continuous 1–12 scaled to [0, 1])
    if add_month:
        moy = (index.month.values / 12.0).astype(np.float32)
        moy_feat = np.tile(moy[:, None, None], (1, N, 1))
        data_list.append(moy_feat)

    X_all = np.concatenate(data_list, axis=-1)
    return X_all
