import torch
import numpy as np
import pandas as pd
from typing import Optional, Sequence


def _parse_dynamic_exog_cols(exog_cols):
    """Parse lag/rolling column names — local copy to avoid sys.modules collisions."""
    lag_cols, roll_cols = {}, {}
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


def recursive_inference_dynamic_exog(
    model: torch.nn.Module,
    target_scaler,
    exog_scaler,
    exog_cols: Sequence[str],
    history_target_unscaled: np.ndarray,   # (lookback,) unscaled past target
    history_exog_unscaled: pd.DataFrame,   # (lookback, len(exog_cols)) unscaled exog
    future_exog_unscaled: pd.DataFrame,    # (horizon, len(exog_cols)) unscaled exog
    target_channel: int = 0,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Recursive 1-step forecast with leak-safe dynamic exogenous features.

    For each forecast step t the function:
      1. Recomputes every `lag_k` column from the running buffer of (past +
         predicted) UNSCALED target values, so the model never sees future
         ground-truth.
      2. Recomputes every `rolling_mean[_excl]_W` column as the mean of the
         last W unscaled target values strictly BEFORE step t.
      3. Scales the row with `exog_scaler` (type-aware: pass-through for
         binary / cyclical, MinMax/Standard for continuous).
      4. Scales the target buffer with `target_scaler`, assembles the model
         input window with the same +1 exog shift as `make_windows`, and
         performs a 1-step forecast.
      5. Inverse-scales the prediction and appends it to the target buffer.

    Parameters
    ----------
    history_target_unscaled : last `lookback` UNSCALED true target values.
    history_exog_unscaled   : exog values aligned with `history_target_unscaled`
                              (used only for the initial input window — its
                              lag/rolling columns are NOT recomputed; they are
                              assumed correct from training-time generation).
    future_exog_unscaled    : exog values for the forecast horizon. Lag/rolling
                              columns will be OVERWRITTEN during inference.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    exog_cols = list(exog_cols)
    horizon = len(future_exog_unscaled)
    # `lookback` is defined by the exog window — target history may be LONGER
    # to support large lags (e.g. lag_364) that need to look back beyond it.
    lookback = len(history_exog_unscaled)

    lag_cols, roll_cols = _parse_dynamic_exog_cols(exog_cols)

    # Running buffer of UNSCALED past targets (history + predictions so far).
    target_buffer = list(np.asarray(history_target_unscaled, dtype=np.float64).ravel())

    # Pre-scale the historical exog block once (lag/rolling there came from
    # ground-truth training data and are valid).
    history_exog_scaled = exog_scaler.transform(
        history_exog_unscaled[exog_cols].copy(), exog_cols
    )
    history_exog_scaled = np.asarray(history_exog_scaled, dtype=np.float32)

    # Only the LAST `lookback` target values seed the input window; the rest
    # of the buffer is kept solely for long-lag lookups.
    recent_target_unscaled = np.asarray(target_buffer[-lookback:], dtype=np.float32)
    history_target_scaled = target_scaler.transform(
        recent_target_unscaled.reshape(-1, 1)
    ).ravel().astype(np.float32)

    # Number of input channels = 1 (target) + len(exog_cols)
    C_in = 1 + len(exog_cols)
    # Indices: by convention target is at index `target_channel` (0).
    exog_indices = [idx for idx in range(C_in) if idx != target_channel]

    # Assemble the initial scaled (lookback, C_in) window with the +1 exog
    # shift used in `make_windows`: the last exog row is the exog for step t.
    initial_window = np.zeros((lookback, C_in), dtype=np.float32)
    initial_window[:, target_channel] = history_target_scaled
    if exog_indices:
        # rows 0..L-2 use history exog rows 1..L-1
        initial_window[:-1, exog_indices] = history_exog_scaled[1:]
        # row L-1 uses exog of step t = first future exog row (computed below)

    future_exog_unscaled = future_exog_unscaled.reset_index(drop=True).copy()

    preds_unscaled = []
    model = model.to(device).eval()
    input_window = initial_window

    with torch.no_grad():
        for i in range(horizon):
            # ------ Step t exog: recompute dynamic columns from target_buffer ------
            row = future_exog_unscaled.iloc[i].copy()
            for col, k in lag_cols.items():
                # lag_k at step t = y_{t-k}; target_buffer ends at y_{t-1}.
                if k <= len(target_buffer):
                    row[col] = target_buffer[-k]
                else:
                    row[col] = 0.0
            for col, w in roll_cols.items():
                # leak-safe rolling mean of the last w UNSCALED targets before t
                window_vals = target_buffer[-w:] if w <= len(target_buffer) else target_buffer
                row[col] = float(np.mean(window_vals)) if window_vals else 0.0

            # Scale this single exog row with the type-aware scaler.
            row_scaled = exog_scaler.transform(
                pd.DataFrame([row[exog_cols].values], columns=exog_cols), exog_cols
            )
            row_scaled = np.asarray(row_scaled, dtype=np.float32).ravel()

            # Place scaled exog into the last input row.
            if exog_indices:
                input_window[-1, exog_indices] = row_scaled

            # ------ Forecast 1 step ------
            x_tensor = torch.from_numpy(input_window).float().unsqueeze(0).to(device)
            y_pred = model(x_tensor)
            val_pred_scaled = float(y_pred.flatten()[0].item())

            # Inverse-scale prediction and store.
            val_pred = float(target_scaler.inverse_transform(
                np.array([[val_pred_scaled]], dtype=np.float32)
            ).ravel()[0])
            preds_unscaled.append(val_pred)
            target_buffer.append(val_pred)

            # ------ Roll the window forward ------
            input_window = np.roll(input_window, -1, axis=0)
            input_window[-1, target_channel] = val_pred_scaled
            # exog of the NEW last row (step t+1) will be filled at next iter

    return np.asarray(preds_unscaled, dtype=np.float32)
