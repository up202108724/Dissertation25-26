"""
Recursive inference for the TCN forecaster with leak-safe dynamic exog.

The TCN (like the MLP) maps (B, L, C) -> (B, 1), so the rollout logic is
identical.  For each forecast step the function:

  1. Recomputes every ``lag_k`` column from the running buffer of (past +
     predicted) UNSCALED target values — no ground-truth test values leak.
  2. Recomputes every ``rolling_mean[_excl]_W`` as the mean of the last W
     UNSCALED target values strictly BEFORE the step being predicted.
  3. Scales the exog row with the type-aware ``exog_scaler``.
  4. Assembles the input window with the same +1 exog shift used in
     ``make_windows`` (the exog row at position t carries the calendar/
     holiday features of the day being predicted at that step).
  5. Feeds the window to the TCN, inverse-scales the output, and appends it
     to the target buffer for the next step's lag/rolling lookups.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ── helpers ───────────────────────────────────────────────────────────────

def parse_dynamic_exog_cols(exog_cols: Sequence[str]):
    """
    Split exog column names into:
      lag_cols  : {col_name: k}   — ``lag_k`` columns
      roll_cols : {col_name: W}   — ``rolling_mean[_excl]_W`` columns

    All other columns are treated as static (calendar / holiday / promo)
    and passed through unchanged.
    """
    lag_cols: dict[str, int] = {}
    roll_cols: dict[str, int] = {}
    for c in exog_cols:
        if c.startswith("lag_"):
            try:
                lag_cols[c] = int(c.split("_")[-1])
            except ValueError:
                pass
        elif c.startswith("rolling_mean_excl_") or c.startswith("rolling_mean_"):
            try:
                roll_cols[c] = int(c.split("_")[-1])
            except ValueError:
                pass
    return lag_cols, roll_cols


# ── main inference entry-point ─────────────────────────────────────────────

@torch.no_grad()
def recursive_inference_dynamic_exog(
    model: nn.Module,
    target_scaler,
    exog_scaler,
    exog_cols: Sequence[str],
    history_target_unscaled: np.ndarray,   # (≥lookback,) unscaled past target
    history_exog_unscaled: pd.DataFrame,   # (lookback, len(exog_cols))
    future_exog_unscaled: pd.DataFrame,    # (horizon,  len(exog_cols))
    target_channel: int = 0,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    One-step-at-a-time recursive forecast for any model with signature
    ``model(x: Tensor[B, L, C]) -> Tensor[B, 1]`` (matches TCNForecaster).

    Parameters
    ----------
    model                   : TCNForecaster (or any compatible model).
    target_scaler           : sklearn scaler fit on the target (e.g. MinMaxScaler).
    exog_scaler             : type-aware ExogenousScaler fit on train exog.
    exog_cols               : ordered list of exogenous column names.
    history_target_unscaled : unscaled target history ending at the last
                              observed day (typically ``concat([train, val])``).
                              Must be long enough to satisfy the largest lag
                              (e.g. lag_364 requires ≥ 364 values).
    history_exog_unscaled   : exog rows aligned to the last ``lookback`` days.
                              Lag/rolling values here are assumed correct
                              (generated from ground-truth training data).
    future_exog_unscaled    : exog rows for the forecast horizon. Lag and
                              rolling-mean columns will be OVERWRITTEN at each
                              step from the running prediction buffer.
    target_channel          : index of the target column in the input tensor
                              (default 0).
    device                  : torch device string; auto-detected if None.

    Returns
    -------
    np.ndarray of shape (horizon,) — inverse-scaled predictions.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    exog_cols = list(exog_cols)
    horizon   = len(future_exog_unscaled)
    lookback  = len(history_exog_unscaled)   # model's input length L

    lag_cols, roll_cols = parse_dynamic_exog_cols(exog_cols)

    # Running buffer of UNSCALED past targets.  May be longer than lookback
    # so that large lags (e.g. lag_364) resolve correctly.
    target_buffer = list(np.asarray(history_target_unscaled, dtype=np.float64).ravel())

    # ── Pre-scale the historical exog block (ground-truth, valid as-is) ──
    history_exog_scaled = exog_scaler.transform(
        history_exog_unscaled[exog_cols].copy(), exog_cols
    )
    history_exog_scaled = np.asarray(history_exog_scaled, dtype=np.float32)

    # ── Scale the target history for the initial input window ─────────────
    recent_target_unscaled = np.asarray(target_buffer[-lookback:], dtype=np.float32)
    history_target_scaled  = target_scaler.transform(
        recent_target_unscaled.reshape(-1, 1)
    ).ravel().astype(np.float32)

    # ── Build (lookback, 1 + n_exog) input window ─────────────────────────
    # Layout: channel 0 = scaled target, channels 1..n_exog = scaled exog.
    # The +1 exog shift: row i carries the exog features of the day whose
    # TARGET appears at row i+1 (i.e., the day being *predicted* at step i).
    #   rows 0..L-2  → history_exog_scaled rows 1..L-1
    #   row L-1      → future_exog step 0  (filled in the first loop iter)
    C_in = 1 + len(exog_cols)
    exog_indices = [j for j in range(C_in) if j != target_channel]

    input_window = np.zeros((lookback, C_in), dtype=np.float32)
    input_window[:, target_channel] = history_target_scaled
    if exog_indices:
        input_window[:-1, exog_indices] = history_exog_scaled[1:]
        # input_window[-1, exog_indices] filled in step 0 of the loop below

    future_exog_unscaled = future_exog_unscaled.reset_index(drop=True).copy()

    preds_unscaled: list[float] = []
    model = model.to(device)
    model.eval()

    for i in range(horizon):
        # ── Step i: recompute dynamic exog from target_buffer ─────────────
        row = future_exog_unscaled.iloc[i].copy()

        # lag_k: look back k steps in the running (unscaled) buffer
        for col, k in lag_cols.items():
            row[col] = target_buffer[-k] if k <= len(target_buffer) else 0.0

        # rolling_mean[_excl]_W: mean of last W unscaled values before step i
        for col, w in roll_cols.items():
            window_vals = target_buffer[-w:] if w <= len(target_buffer) else target_buffer
            row[col] = float(np.mean(window_vals)) if window_vals else 0.0

        # Scale the single exog row
        row_scaled = exog_scaler.transform(
            pd.DataFrame([row[exog_cols].values], columns=exog_cols), exog_cols
        )
        row_scaled = np.asarray(row_scaled, dtype=np.float32).ravel()

        if exog_indices:
            input_window[-1, exog_indices] = row_scaled

        # ── Forward pass ──────────────────────────────────────────────────
        x_t = torch.from_numpy(input_window).float().unsqueeze(0).to(device)
        y_hat_scaled = float(model(x_t).flatten()[0].item())

        # ── Inverse-scale and store ───────────────────────────────────────
        y_hat = float(
            target_scaler.inverse_transform(
                np.array([[y_hat_scaled]], dtype=np.float32)
            ).ravel()[0]
        )
        preds_unscaled.append(y_hat)
        target_buffer.append(y_hat)   # feed own prediction for next lags

        # ── Roll the input window forward ─────────────────────────────────
        input_window = np.roll(input_window, -1, axis=0)
        input_window[-1, target_channel] = y_hat_scaled
        # exog for the new last row is set at the top of the next iteration

    return np.asarray(preds_unscaled, dtype=np.float32)
