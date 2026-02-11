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
