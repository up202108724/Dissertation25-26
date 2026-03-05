import numpy as np
from typing import Tuple

def make_windows(
    series: np.ndarray,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:

    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 1:
        series = series[:, None]  # (T, 1)

    T, C = series.shape
    N = T - lookback - horizon + 1
    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    X = np.zeros((N, lookback, C), dtype=np.float32)
    y = np.zeros((N, horizon, C), dtype=np.float32)

    for i in range(N):
        X[i] = series[i : i + lookback]
        y[i] = series[i + lookback : i + lookback + horizon]

    return X, y
