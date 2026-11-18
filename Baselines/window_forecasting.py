"""
Shared one-step window training + recursive rollout for the non-LSTM baselines
(DLinearBaseline, PatchTSTBaseline, and the neighbour baselines in
GCN_GAT_Graph2Vec_Assemble_InferenceFix/review_baselines.py).

Same protocol as the graph hybrids: one-step-ahead training on lookback
windows, validation windows from the validation slice only, AdamW + MSE +
grad-clip 1.0 + early stopping that restores the best-validation weights, and
a recursive rollout seeded with the last ``L`` validation days in which lag /
rolling-mean exog columns are recomputed from the model's own forecasts.
"""

from __future__ import annotations

import copy
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

LAG_K_BY_NAME = {"lag_1": 1, "lag_7": 7, "lag_30": 30}
ROLLING_MEAN_EXCL_PREFIX = "rolling_mean_excl_"


class WindowDataset(Dataset):
    """Lookback windows ``[target ‖ exog ‖ extra]`` → next target value.

    exog follows ``LSTMBaseline.dataset`` (row k carries the exog of day k+1,
    i.e. the day being predicted at the last row); extra (e.g. neighbour)
    channels are aligned with the target day of each row.
    """

    def __init__(self, target, exog=None, extra=None, lookback: int = 30):
        self.target = np.asarray(target, dtype=np.float32)
        self.exog = None if exog is None else np.asarray(exog, dtype=np.float32)
        self.extra = None if extra is None else np.asarray(extra, dtype=np.float32)
        self.L = int(lookback)

    def __len__(self):
        return len(self.target) - self.L

    def __getitem__(self, idx):
        L = self.L
        cols = [self.target[idx:idx + L, None]]
        if self.exog is not None:
            cols.append(self.exog[idx + 1:idx + L + 1])
        if self.extra is not None:
            cols.append(self.extra[idx:idx + L])
        x = np.concatenate(cols, axis=1)
        return torch.from_numpy(x), torch.tensor([self.target[idx + L]])


def dynamic_exog_indices(exog_cols: Optional[Sequence[str]]):
    """``({col: k} for lag_k, {col: W} for rolling_mean_excl_W})`` within exog."""
    lag, roll = {}, {}
    for i, name in enumerate(exog_cols or []):
        if name in LAG_K_BY_NAME:
            lag[i] = LAG_K_BY_NAME[name]
        elif name.startswith(ROLLING_MEAN_EXCL_PREFIX):
            try:
                roll[i] = int(name[len(ROLLING_MEAN_EXCL_PREFIX):])
            except ValueError:
                pass
    return lag, roll


def scale_lag_value(raw_value: float, exog_scaler, col_idx: int) -> float:
    """Scale one scalar through column ``col_idx`` of a fitted sklearn scaler
    (same as ``inference_strategies._scale_lag_value``)."""
    if exog_scaler is None:
        return float(raw_value)
    row = np.zeros((1, exog_scaler.n_features_in_), dtype=np.float32)
    row[0, col_idx] = float(raw_value)
    return float(exog_scaler.transform(row)[0, col_idx])


def reseed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_window_model(model, train_loader, val_loader, *, lr, epochs, patience,
                       device, use_scheduler=True, grad_clip=1.0):
    """Same loop as ``train_methods.train_gcn_*``: AdamW, MSE, clip, early stop,
    best-validation weights restored in memory."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=max(1, patience // 3))
        if use_scheduler else None)

    train_losses, val_losses = [], []
    best_val, best_epoch, best_state, bad = float('inf'), 0, None, 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).view(-1), yb.view(-1))
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += loss.item()
        train_losses.append(total / max(1, len(train_loader)))

        model.eval()
        outs, ys = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                outs.append(model(xb.to(device)).view(-1))
                ys.append(yb.to(device).view(-1))
        val_loss = criterion(torch.cat(outs), torch.cat(ys)).item()
        val_losses.append(val_loss)
        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, best_epoch, bad = val_loss, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_epoch


@torch.no_grad()
def recursive_forecast_window_model(model, *, seed_rows, scaler, horizon, device,
                                    n_exog=0, exog_test=None, extra_future=None,
                                    target_history_unscaled=None,
                                    lag_col_indices=None,
                                    rolling_mean_excl_col_indices=None,
                                    exog_scaler=None):
    """Recursive rollout matching ``_recursive_forecast_gcn_*_perstep``.

    seed_rows    : (L, 1 + n_exog + n_extra) scaled rows; the last row already
                   holds the exog of the first forecast day.
    extra_future : (horizon, n_extra) scaled extra values for each forecast
                   day; row ``s`` enters the window after day ``s`` is predicted.
    """
    model.eval()
    ts = np.asarray(seed_rows, dtype=np.float32).copy()
    ex0, ex1 = 1, 1 + n_exog
    lag_col_indices = lag_col_indices or {}
    rolling_mean_excl_col_indices = rolling_mean_excl_col_indices or {}
    dyn = (bool(lag_col_indices) or bool(rolling_mean_excl_col_indices)) \
        and target_history_unscaled is not None and n_exog > 0
    if dyn:
        if exog_scaler is None:
            raise ValueError("exog_scaler is required for lag / rolling exog columns")
        y_hist = np.asarray(target_history_unscaled, dtype=np.float32).copy()

    preds = []
    for step in range(horizon):
        if step > 0 and exog_test is not None and n_exog > 0:
            ts[-1, ex0:ex1] = exog_test[step]
        if dyn:
            for col, k in lag_col_indices.items():
                ts[-1, ex0 + col] = scale_lag_value(float(y_hist[-k]), exog_scaler, col)
            for col, W in rolling_mean_excl_col_indices.items():
                ts[-1, ex0 + col] = scale_lag_value(float(np.mean(y_hist[-W:])), exog_scaler, col)

        y_hat = float(model(torch.from_numpy(ts).unsqueeze(0).to(device)).view(-1)[-1].item())
        preds.append(y_hat)

        ts = np.vstack([ts[1:], ts[-1:].copy()])
        ts[-1, 0] = y_hat
        if extra_future is not None:
            ts[-1, ex1:] = extra_future[step]
        if dyn:
            y_hist = np.append(
                y_hist, scaler.inverse_transform(np.array([[y_hat]], dtype=np.float32))[0, 0])

    return scaler.inverse_transform(np.asarray(preds, dtype=np.float32).reshape(-1, 1)).flatten()


def fit_and_forecast(model, *, lookback, train_scaled, val_scaled, scaler, horizon,
                     device, epochs, patience, batch_size, lr, use_scheduler=True,
                     exog_train=None, exog_val=None, exog_test=None,
                     exog_scaler=None, exog_cols=None,
                     extra_train=None, extra_val=None, extra_future=None,
                     target_history_unscaled=None):
    """Train ``model`` (``forward(x: (B, L, C))``) on the product's windows and
    roll it out recursively.  Returns ``(forecast_raw_units, train_losses,
    val_losses)``.  Call :func:`reseed` before building ``model``."""
    model = model.to(device)
    n_exog = 0 if exog_train is None else exog_train.shape[1]
    n_extra = 0 if extra_train is None else extra_train.shape[1]
    train_loader = DataLoader(WindowDataset(train_scaled, exog_train, extra_train, lookback),
                              batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(WindowDataset(val_scaled, exog_val, extra_val, lookback),
                            batch_size=batch_size, shuffle=False)
    model, t_losses, v_losses, _ = train_window_model(
        model, train_loader, val_loader, lr=lr, epochs=epochs, patience=patience,
        device=device, use_scheduler=use_scheduler)

    cols = [np.asarray(val_scaled[-lookback:], dtype=np.float32)[:, None]]
    if n_exog:
        cols.append(np.vstack([exog_val[-(lookback - 1):], exog_test[0:1]]))
    if n_extra:
        cols.append(extra_val[-lookback:])
    seed_rows = np.concatenate(cols, axis=1)

    lag_idx, roll_idx = dynamic_exog_indices(exog_cols)
    forecast = recursive_forecast_window_model(
        model, seed_rows=seed_rows, scaler=scaler, horizon=horizon, device=device,
        n_exog=n_exog, exog_test=exog_test, extra_future=extra_future,
        target_history_unscaled=target_history_unscaled,
        lag_col_indices=lag_idx, rolling_mean_excl_col_indices=roll_idx,
        exog_scaler=exog_scaler)
    return forecast, t_losses, v_losses
