"""
inference.py — Recursive 1-step-ahead inference for StaticGCNLSTMForecaster.

At each inference step:
  1. The global graph (prebuilt, fixed topology + fixed node features) is
     passed to the model exactly as during training.
  2. The target node's GCN embedding is extracted and used to init LSTM h₀/c₀.
  3. The LSTM processes the rolling lookback window and produces the next step.
  4. The rolling window is updated with the new prediction and we repeat.

No graph is rebuilt at inference time — the static graph is reused as-is.
"""

import numpy as np
import torch
from typing import Optional


def recursive_inference(
    model: torch.nn.Module,
    scaler,
    graph_data,                   # PyG Data from build_static_graph (on device)
    target_node_idx: int,         # index of the target item in the global graph
    recent_history: np.ndarray,   # (lookback, 1 + cal_dim) — raw unscaled target + raw exog
    future_exog: np.ndarray,      # (horizon, cal_dim) — exog for each forecast step
    target_channel: int = 0,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Recursive horizon-step forecast using the static-graph GCN+LSTM model.

    Parameters
    ----------
    model            : trained StaticGCNLSTMForecaster
    scaler           : fitted scaler for the target channel
    graph_data       : PyG Data object with x, edge_index, edge_weight
    target_node_idx  : node index for the target item
    recent_history   : (lookback, 1+cal_dim)  raw values (target + exog)
    future_exog      : (horizon, cal_dim)  pre-scaled exog for each future step
    target_channel   : column index of the target in recent_history
    device           : 'cpu' or 'cuda'

    Returns
    -------
    np.ndarray of shape (horizon,)  — unscaled forecast values
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev    = torch.device(device)

    model  = model.to(dev).eval()
    gx     = graph_data.x.to(dev)
    ei     = graph_data.edge_index.to(dev)
    ew     = graph_data.edge_weight.to(dev) if graph_data.edge_weight.numel() > 0 else None

    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    lookback = recent_history.shape[0]
    cal_dim  = recent_history.shape[1] - 1
    horizon  = len(future_exog)

    # Scale target channel from raw history
    x_scaled = recent_history.copy()
    x_scaled[:, target_channel:target_channel+1] = scaler.transform(
        recent_history[:, target_channel:target_channel+1]
    )

    # Rolling windows (scaled)
    target_window = x_scaled[:, target_channel].copy()   # (lookback,)
    cal_window    = x_scaled[:, 1:].copy() if cal_dim > 0 else None   # (lookback, cal_dim)

    idx_tensor = torch.tensor([target_node_idx], dtype=torch.long, device=dev)
    preds_unscaled = []

    with torch.no_grad():
        for i in range(horizon):
            cal_next = (
                future_exog[i].astype(np.float32)
                if future_exog.ndim > 1
                else np.array([future_exog[i]], dtype=np.float32)
            )  # (cal_dim,)

            # Build ts_seq: (1, lookback, 1+cal_dim)
            if cal_dim > 0:
                cal_shifted = np.vstack([
                    cal_window[1:],          # rows 1..lookback-1
                    cal_next.reshape(1, -1), # new next-step cal
                ])  # (lookback, cal_dim)
                ts_np = np.concatenate(
                    [target_window[:, None], cal_shifted], axis=1
                )  # (lookback, 1+cal_dim)
            else:
                ts_np = target_window[:, None]   # (lookback, 1)

            ts_t = torch.from_numpy(ts_np).float().unsqueeze(0).to(dev)   # (1, lookback, 1+cal_dim)

            pred_scaled = model(gx, ei, ew, idx_tensor, ts_t)
            val_scaled  = pred_scaled.reshape(-1)[0].item()

            unscaled = scaler.inverse_transform([[val_scaled]])[0, 0]
            preds_unscaled.append(unscaled)

            # Shift rolling windows
            target_window = np.roll(target_window, -1)
            target_window[-1] = val_scaled
            if cal_dim > 0:
                cal_window = np.roll(cal_window, -1, axis=0)
                cal_window[-1] = cal_next

    return np.array(preds_unscaled, dtype=np.float32)
