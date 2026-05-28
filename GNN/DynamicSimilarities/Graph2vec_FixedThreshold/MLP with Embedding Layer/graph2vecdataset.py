from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from typing import Tuple

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, emb: np.ndarray, y: np.ndarray):
        self.X   = torch.from_numpy(X)    # (N, L, C)
        self.emb = torch.from_numpy(emb)  # (N, L, emb_dim)  or (N, L, 0)
        self.y   = torch.from_numpy(y)    # (N, H, C)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.emb[idx], self.y[idx]


def make_windows(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    target_channel: int = 0,
    embeddings: np.ndarray = None,
    graph_window_size: int = 7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X   : (N, lookback, C)        — target + calendar features, NO embeddings
    emb : (N, lookback, emb_dim)  — raw Graph2Vec embeddings (zeros when None)
    y   : (N, horizon, C)         — forecast targets

    Embeddings are returned as a *separate* array so the model can pass them
    through a trainable projection layer rather than treating them as plain
    flat features.
    """
    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 1:
        series = series[:, None]  # (T, 1)

    T, C = series.shape
    N = T - lookback - horizon + 1
    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    # Pad embeddings with leading zeros (same convention as the LSTM dataset)
    if embeddings is not None:
        emb_dim = embeddings.shape[1] if embeddings.ndim > 1 else 1
        if len(embeddings) < T:
            zero_pad = np.zeros((graph_window_size, emb_dim), dtype=np.float32)
            padded_embeddings = np.vstack([zero_pad, embeddings]).astype(np.float32)
        else:
            padded_embeddings = np.asarray(embeddings, dtype=np.float32)
    else:
        emb_dim = 0
        padded_embeddings = None

    X   = np.zeros((N, lookback, C),       dtype=np.float32)
    emb = np.zeros((N, lookback, emb_dim), dtype=np.float32)
    y   = np.zeros((N, horizon, C),        dtype=np.float32)

    exog_indices = [idx for idx in range(C) if idx != target_channel]

    for i in range(N):
        window_base = series[i : i + lookback].copy()

        # Shift exogenous variables forward by 1 so the model sees the target
        # day's calendar features (same convention as training)
        if exog_indices:
            window_base[:, exog_indices] = series[i + 1 : i + lookback + 1, exog_indices]

        X[i] = window_base

        if padded_embeddings is not None:
            emb[i] = padded_embeddings[i : i + lookback]

        y[i] = series[i + lookback : i + lookback + horizon]

    return X, emb, y