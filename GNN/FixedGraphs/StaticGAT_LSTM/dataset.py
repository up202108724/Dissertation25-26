"""
dataset.py — Dataset utilities for the Static-GCN + LSTM forecaster.

Key difference from the per-sample ego-graph approach:
  • No per-sample graph is built here.
  • Each sample stores only (ts_seq, target_node_idx, y).
  • The single global graph is passed separately to the model at train time.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StaticGCNDataset(Dataset):
    """
    Each sample:
        ts_seq          – (lookback, 1 + cal_dim)  LSTM input sequence
        target_node_idx – int  index into the global graph's node list
        y               – (horizon, 1)  target values to predict
    """

    def __init__(
        self,
        ts_seqs: np.ndarray,          # (N, lookback, 1+cal_dim)
        target_node_indices: np.ndarray,   # (N,) int
        y: np.ndarray,                # (N, horizon, 1)
    ):
        self.ts_seqs             = torch.from_numpy(ts_seqs.astype(np.float32))
        self.target_node_indices = torch.from_numpy(target_node_indices.astype(np.int64))
        self.y                   = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (
            self.ts_seqs[idx],
            self.target_node_indices[idx],
            self.y[idx],
        )


# ---------------------------------------------------------------------------
# Sliding-window builder
# ---------------------------------------------------------------------------

def make_windows(
    series: np.ndarray,       # (T, 1)   scaled target values
    cal: np.ndarray,          # (T, cal_dim) scaled calendar features
    lookback: int,
    horizon: int,
    target_node_idx: int,
) -> tuple:
    """
    Build sliding-window samples for one item.

    ts_seq[t] = (value_t, cal_{t+1})   — calendar is shifted forward by 1
    so that the LSTM sees the calendar features of the day being predicted.

    Parameters
    ----------
    series          : (T, 1)  scaled target
    cal             : (T, cal_dim)  scaled calendar
    lookback        : sequence length fed to LSTM
    horizon         : steps ahead to predict
    target_node_idx : node index in the global graph for this item

    Returns
    -------
    ts_seqs  : (N, lookback, 1+cal_dim)
    indices  : (N,) int  – all equal to target_node_idx
    y        : (N, horizon, 1)
    """
    series = np.asarray(series, dtype=np.float32)
    cal    = np.asarray(cal,    dtype=np.float32)

    if series.ndim == 1:
        series = series[:, None]

    T       = series.shape[0]
    cal_dim = cal.shape[1] if cal.ndim == 2 else 0
    N       = T - lookback - horizon + 1

    if N <= 0:
        raise ValueError(
            f"Time series too short: T={T}, lookback={lookback}, horizon={horizon}"
        )

    ts_seqs = np.zeros((N, lookback, 1 + cal_dim), dtype=np.float32)
    y_out   = np.zeros((N, horizon, 1),            dtype=np.float32)

    for i in range(N):
        # Target values for the lookback window
        ts_window = series[i : i + lookback, 0]   # (lookback,)

        if cal_dim > 0:
            # Calendar shifted by +1: at position t show cal of day t+1
            cal_shifted = cal[i + 1 : i + lookback + 1]  # (lookback, cal_dim)
            ts_seqs[i] = np.column_stack([ts_window, cal_shifted])
        else:
            ts_seqs[i, :, 0] = ts_window

        y_out[i, :, 0] = series[i + lookback : i + lookback + horizon, 0]

    indices = np.full(N, target_node_idx, dtype=np.int64)
    return ts_seqs, indices, y_out
