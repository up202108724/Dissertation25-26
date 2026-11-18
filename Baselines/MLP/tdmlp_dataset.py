from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, L, C)
        self.y = torch.from_numpy(y)  # (N, H, C)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_windows(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    target_channel: int = 0,
    embeddings: Optional[np.ndarray] = None,
    graph_window_size: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) lookback windows for the TDMLP.

    Baseline: ``embeddings=None`` -> X is ``[target | exog]``.
    Graph2vec: pass ``embeddings`` (and ``graph_window_size`` = the graph
    sliding window W) -> X is ``[target | exog | graph embedding]``.
    """
    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 1:
        series = series[:, None]  # (T, 1)

    T, C = series.shape
    N = T - lookback - horizon + 1
    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    has_embeddings = embeddings is not None
    if has_embeddings:
        # Lazy import: baseline runs don't pull in the Graph2vec stack.
        from Graph2vec.generate_graph2vecwithadaptativethreshold import pad_graph_embeddings

        # Same padding as the LSTM TimeSeriesDataset
        padded_embeddings = pad_graph_embeddings(embeddings, len(series), graph_window_size)
        out_C = C + padded_embeddings.shape[1]
    else:
        out_C = C

    X = np.zeros((N, lookback, out_C), dtype=np.float32)
    y = np.zeros((N, horizon, C), dtype=np.float32)

    exog_indices = [idx for idx in range(C) if idx != target_channel]

    for i in range(N):
        window_base = series[i : i + lookback].copy()

        # Shift exogenous variables forward by 1 so the model sees the target day's exog features
        # X[i] target corresponds to steps i ... i+lookback-1
        # X[i] exog corresponds to steps i+1 ... i+lookback
        if exog_indices:
            window_base[:, exog_indices] = series[i + 1 : i + lookback + 1, exog_indices]

        if has_embeddings:
            X[i] = np.column_stack([window_base, padded_embeddings[i : i + lookback]])
        else:
            X[i] = window_base

        y[i] = series[i + lookback : i + lookback + horizon]

    return X, y
