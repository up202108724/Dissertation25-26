"""
Optional torch Dataset/DataLoader wrappers for the global MTGNN forecaster.

These produce the exact tensor contract MTGNN expects:
    X : (seq_len, N, C)   per-node demand ‖ exogenous channels
    Y : (horizon, N, 1)   per-node demand target

NOTE: ``mtgnn_pipeline.py`` does NOT use this module — it builds windows with
``make_windows`` and applies per-node MinMax scaling + the shared calendar exog
itself (so train-only scaling and the recursive roll-out stay consistent).
This module is provided for callers who prefer the standard DataLoader idiom;
feed it dataframes that are ALREADY scaled, since no scaling is done here.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MTGNNDataset(Dataset):
    def __init__(self, df, seq_len=30, horizon=1, target_col="value", feature_cols=None):
        """
        df : long-format dataframe with columns ['date', 'item_id', target_col]
             (+ feature_cols). All N products, sorted by date.
        """
        self.seq_len = seq_len
        self.horizon = horizon

        # (T, N) demand matrix, products aligned chronologically
        target_df = df.pivot(index="date", columns="item_id", values=target_col).fillna(0.0)
        self.y_data = target_df.values.astype(np.float32)

        # (T, N, C) input channels — demand first, then any exogenous features
        self.features = feature_cols if feature_cols else [target_col]
        self.X_data = np.stack(
            [df.pivot(index="date", columns="item_id", values=f).fillna(0.0).values
             for f in self.features],
            axis=-1,
        ).astype(np.float32)

        self.num_samples = len(self.X_data) - seq_len - horizon + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X = self.X_data[idx: idx + self.seq_len]                         # (L, N, C)
        Y = self.y_data[idx + self.seq_len: idx + self.seq_len + self.horizon]
        if Y.ndim == 2:
            Y = np.expand_dims(Y, axis=-1)                               # (H, N, 1)
        return torch.from_numpy(X), torch.from_numpy(Y)


def create_mtgnn_dataloaders(train_df, val_df, test_df, feature_cols,
                             seq_len=30, horizon=1, batch_size=32, target_col="value"):
    """Build train/val/test DataLoaders. Inputs must already be scaled."""
    train_ds = MTGNNDataset(train_df, seq_len, horizon, target_col, feature_cols)
    val_ds   = MTGNNDataset(val_df,   seq_len, horizon, target_col, feature_cols)
    test_ds  = MTGNNDataset(test_df,  seq_len, horizon, target_col, feature_cols)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    )
