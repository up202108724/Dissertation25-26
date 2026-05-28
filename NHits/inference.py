import torch
import numpy as np
import pandas as pd
from typing import Optional, Sequence


def recursive_inference_nhits(
    model: torch.nn.Module,
    scaler,
    recent_history: np.ndarray,
    future_exog: np.ndarray,
    target_channel: int = 0,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Recursive 1-step forecast for an ``NHITS`` model trained with horizon=1.

    The NHITS forward signature is ``model(x, future_exog) -> (B, H, 1)`` —
    i.e. it takes the lookback tensor PLUS a separate ``(B, H, n_exog)``
    future-exog tensor.  For horizon=1 recursion we extract only the FIRST
    future-exog row for each step (the row corresponding to the day being
    predicted) and feed it as a ``(1, 1, n_exog)`` tensor.

    Parameters
    ----------
    model            : trained NHITS instance (horizon=1)
    scaler           : sklearn scaler fit on the target column
    recent_history   : (lookback, 1 + n_exog) — column 0 = scaled-or-unscaled
                       target, columns 1: = ALREADY SCALED exog rows aligned
                       with the lookback days.  The target column is scaled
                       inside this function if it is not yet scaled (we always
                       call ``scaler.transform`` defensively).
    future_exog      : (horizon, n_exog) ALREADY SCALED exogenous values for
                       each forecast step.  Assumed to be static w.r.t. the
                       target (calendar / holiday); use a dynamic-exog variant
                       if lag_*/rolling_mean_* features are present.
    target_channel   : index of the target column in ``recent_history``.
    device           : torch device (defaults to cuda if available).

    Returns
    -------
    np.ndarray of shape (horizon,) — INVERSE-SCALED predictions.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    horizon = len(future_exog)

    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]
    future_exog = np.asarray(future_exog, dtype=np.float32)
    if future_exog.ndim == 1:
        future_exog = future_exog[:, None]

    C_in = recent_history.shape[1]
    n_exog = C_in - 1
    exog_indices = [idx for idx in range(C_in) if idx != target_channel]

    # Scale only the target column (the exog columns are expected to already
    # be scaled by the caller, matching the convention used in main.py).
    current_x_scaled = recent_history.copy()
    current_x_scaled[:, target_channel:target_channel + 1] = scaler.transform(
        recent_history[:, target_channel:target_channel + 1]
    )

    preds_scaled = []
    model = model.to(device).eval()
    input_window = current_x_scaled.copy()

    with torch.no_grad():
        for i in range(horizon):
            x_tensor = torch.from_numpy(input_window).float().unsqueeze(0).to(device)  # (1, L, C)
            if n_exog > 0:
                fut_tensor = (
                    torch.from_numpy(future_exog[i : i + 1])
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )  # (1, 1, n_exog)
                y_pred = model(x_tensor, fut_tensor)
            else:
                y_pred = model(x_tensor)

            # NHITS returns (B, H, 1); horizon=1 -> scalar
            val_pred = float(y_pred.reshape(-1)[0].item())
            preds_scaled.append(val_pred)

            # Roll the lookback window forward: drop oldest row, append a new
            # row carrying (ŷ, exog_for_next_step) so the next forward sees a
            # coherent (target ‖ exog) context.
            input_window = np.roll(input_window, -1, axis=0)
            input_window[-1, target_channel] = val_pred
            if exog_indices and (i + 1) < horizon:
                input_window[-1, exog_indices] = future_exog[i + 1]

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()


