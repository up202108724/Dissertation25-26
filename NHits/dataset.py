from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

def make_direct_windows(
    target: np.ndarray,
    exog: np.ndarray,
    lookback: int,
    horizon: int,
):
    """
    Build direct multi-horizon windows.

    Parameters
    ----------
    target   : (T,) scaled target
    exog     : (T, n_exog) scaled exog (may be empty with shape (T, 0))
    lookback : L
    horizon  : H

    Returns
    -------
    X         : (N, L, 1 + n_exog) lookback (target ‖ exog)
    fut_exog  : (N, H, n_exog)    known exog for the forecast window
    y         : (N, H, 1)          forecast target
    """
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    exog   = np.asarray(exog,   dtype=np.float32)
    if exog.ndim == 1:
        exog = exog[:, None]
    T = target.shape[0]
    if exog.shape[0] != T:
        raise ValueError("target and exog must have matching length")

    N = T - lookback - horizon + 1
    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    n_exog = exog.shape[1]
    X        = np.zeros((N, lookback, 1 + n_exog), dtype=np.float32)
    fut_exog = np.zeros((N, horizon, n_exog),      dtype=np.float32)
    y        = np.zeros((N, horizon, 1),           dtype=np.float32)

    for i in range(N):
        X[i, :, 0]   = target[i : i + lookback]
        if n_exog:
            X[i, :, 1:] = exog[i : i + lookback]
            fut_exog[i] = exog[i + lookback : i + lookback + horizon]
        y[i, :, 0]   = target[i + lookback : i + lookback + horizon]

    return X, fut_exog, y