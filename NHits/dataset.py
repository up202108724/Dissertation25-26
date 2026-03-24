from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, L, C)
        self.y = torch.from_numpy(y)  # (N, H, C)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]