from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Tuple, Dict

class WindowDatasetWithCategorical(Dataset):
    def __init__(self, X_cont: np.ndarray, X_cat: Dict[str, np.ndarray], y: np.ndarray):
        
        self.X_cont = torch.from_numpy(X_cont).float()
        self.X_cat = {k: torch.from_numpy(v).long() for k, v in X_cat.items()}
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X_cont.shape[0]

    def __getitem__(self, idx):
        cat_item = {k: v[idx] for k, v in self.X_cat.items()}
        return self.X_cont[idx], cat_item, self.y[idx]


def make_windows_with_categorical(
    series_cont: np.ndarray,           # Continuous features including target (T, C_cont)
    series_cat: Dict[str, np.ndarray], # Categorical features {name: (T, )}
    lookback: int,
    horizon: int,
    target_channel: int = 0
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:

    series_cont = np.asarray(series_cont, dtype=np.float32)
    if series_cont.ndim == 1:
        series_cont = series_cont[:, None]

    T, C_cont = series_cont.shape
    N = T - lookback - horizon + 1
    if N <= 0:
        raise ValueError("Time series too short for given lookback/horizon.")

    X_cont = np.zeros((N, lookback, C_cont), dtype=np.float32)
    y = np.zeros((N, horizon, 1), dtype=np.float32)
    
    X_cat = {k: np.zeros((N, lookback), dtype=np.int64) for k in series_cat.keys()}

    exog_indices = [idx for idx in range(C_cont) if idx != target_channel]

    for i in range(N):
        X_cont[i] = series_cont[i : i + lookback].copy()
        
        # Shift continuous exogenous factors forward by 1
        if len(exog_indices) > 0:
            X_cont[i, :, exog_indices] = series_cont[i + 1 : i + lookback + 1, exog_indices]
            
        # Shift categorical exogenous factors forward by 1
        for cat_name, v_series in series_cat.items():
            v_series = np.asarray(v_series).flatten()
            X_cat[cat_name][i, :] = v_series[i + 1 : i + lookback + 1]
            
        y[i] = series_cont[i + lookback : i + lookback + horizon, target_channel : target_channel + 1]

    return X_cont, X_cat, y

