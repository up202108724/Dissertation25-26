"""
Recursive (one-step-at-a-time) inference for the PER-STEP GCN + TCN forecaster.

The model is fully agnostic to the temporal head (LSTM / MLP / TCN) — the only
contract is the forward signature ``(pyg_batch, target_node_indices, ts_seq)
-> (B, H, 1)``.  At inference we maintain a rolling deque of length L of
ego-graphs alongside the L-row temporal window and recurse forecast-by-forecast:

  1. Run the model on the current (L-graph deque, L-row ts_seq).
  2. Roll the temporal window forward (drop oldest target value, append ŷ;
     advance the exogenous row to the step we are about to predict).
  3. Roll the graph deque forward: drop the oldest graph, append a new
     "current" graph from the per-day-aligned ``future_graphs`` sequence
     (or reuse the last graph if none was provided for the step).

Optionally, exogenous columns whose value at day t is derived from past
target observations (lag_k, rolling_mean_excl_W) are recomputed from the
model's own rolling forecast at every step so the test-time series is
never contaminated by ground-truth target values.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
from torch_geometric.data import Batch, Data


# ──────────────────────────────────────────────────────────────────────────
# Per-day alignment helper
# ──────────────────────────────────────────────────────────────────────────
def _make_pad_graph(template: Data) -> Data:
    """Zero-valued single-node graph matching the feature width of ``template``."""
    in_feats = template.x.shape[1]
    return Data(
        x=torch.zeros(1, in_feats, dtype=torch.float32),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        edge_attr=torch.zeros(1, 1, dtype=torch.float32),
        num_nodes=1,
    )


def _align_pyg_windows_to_timeline(pyg_windows, window_size, step_size, T):
    """
    Convert the per-window PyG list (one Data per sliding window of width W)
    into a per-day list of length T where ``aligned[t]`` is the graph built
    *strictly before* day t (days t-W .. t-1).
    """
    if step_size != 1:
        raise NotImplementedError("Per-day alignment helper currently assumes step_size=1")

    pad = _make_pad_graph(pyg_windows[0])
    aligned = [pad] * window_size + list(pyg_windows)       # pad by W (not W-1)
    if len(aligned) < T:
        aligned += [aligned[-1]] * (T - len(aligned))
    else:
        aligned = aligned[:T]
    return aligned


# ──────────────────────────────────────────────────────────────────────────
# Dynamic-exog helper
# ──────────────────────────────────────────────────────────────────────────
def _scale_lag_value(raw_value: float, exog_scaler, col_idx: int) -> float:
    """
    Apply a single-column MinMaxScaler / StandardScaler transform on one scalar.
    Embeds the scalar into a zero row, transforms, returns the same column.
    """
    if exog_scaler is None:
        return float(raw_value)
    n_features = getattr(exog_scaler, "n_features_in_", None)
    if n_features is None:
        ref = getattr(exog_scaler, "data_min_", None)
        if ref is None:
            ref = getattr(exog_scaler, "mean_", None)
        if ref is None:
            raise ValueError("exog_scaler must expose n_features_in_, data_min_ or mean_")
        n_features = len(ref)
    row = np.zeros((1, n_features), dtype=np.float32)
    row[0, col_idx] = float(raw_value)
    return float(exog_scaler.transform(row)[0, col_idx])


# ──────────────────────────────────────────────────────────────────────────
# Recursive inference (PER-STEP: deque of L graphs)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _recursive_forecast_gcn_perstep(model, ts_seed, initial_graphs,
                                    future_graphs, exog_test_scaled,
                                    scaler, horizon, device,
                                    target_history_unscaled=None,
                                    lag_col_indices: Optional[Dict[int, int]] = None,
                                    rolling_mean_excl_col_indices: Optional[Dict[int, int]] = None,
                                    exog_scaler=None):
    """
    One-step-at-a-time inference for the per-step GCN + TCN (or any other
    per-step temporal head sharing the same forward signature).

    Parameters
    ----------
    model            : SimpleGCNTCNForecaster (per-step variant)
    ts_seed          : (L, 1+n_exog) scaled temporal seed; row L-1 already
                       contains the exog for the FIRST predicted step.
    initial_graphs   : list of L Data ego-graphs (oldest..newest) aligned to
                       the L lookback days at inference start.
    future_graphs    : list of length ``horizon`` of ego-graphs aligned to
                       each successive forecast day.
    exog_test_scaled : (horizon, n_exog) scaled exog for the test window.
    scaler           : sklearn MinMaxScaler fit on the target.
    horizon          : number of recursive steps.
    device           : torch.device

    Leakage-safe dynamic exog (optional — pass ``target_history_unscaled``
    + ``exog_scaler`` and at least one of the col-index dicts to activate):
    target_history_unscaled : 1-D np.ndarray with the RAW (unscaled) target
        history that ends at the day immediately BEFORE the first forecast
        step.  Extended at each step with the model's own (inverse-scaled)
        prediction so that subsequent lookups never peek at ground-truth.
    lag_col_indices : mapping ``{col_in_exog -> k}``.  At every step the
        injected row is overwritten with ``scale(target_history_unscaled[-k])``.
    rolling_mean_excl_col_indices : mapping ``{col_in_exog -> W}`` for
        ``rolling_mean_excl_W`` features.
    exog_scaler     : the sklearn scaler used to scale the exog matrix.

    Returns
    -------
    np.ndarray (horizon,) — inverse-scaled predictions.
    """
    model.eval()
    ts = np.asarray(ts_seed, dtype=np.float32).copy()
    L = ts.shape[0]
    if len(initial_graphs) != L:
        raise ValueError(f"initial_graphs must have length L={L}, got {len(initial_graphs)}")

    lag_col_indices = lag_col_indices or {}
    rolling_mean_excl_col_indices = rolling_mean_excl_col_indices or {}
    use_dynamic_exog = (
        (len(lag_col_indices) > 0 or len(rolling_mean_excl_col_indices) > 0)
        and target_history_unscaled is not None
    )
    if use_dynamic_exog:
        if exog_scaler is None:
            raise ValueError(
                "exog_scaler is required when lag_col_indices or "
                "rolling_mean_excl_col_indices is provided."
            )
        y_history = np.asarray(target_history_unscaled, dtype=np.float32).copy()
        max_window = max(
            list(lag_col_indices.values()) +
            list(rolling_mean_excl_col_indices.values()),
            default=0,
        )
        if len(y_history) < max_window:
            raise ValueError(
                f"target_history_unscaled has {len(y_history)} entries but "
                f"requires at least {max_window} (max of lags / rolling windows)."
            )

    graphs: "deque[Data]" = deque((g.clone() for g in initial_graphs), maxlen=L)
    preds_scaled = []

    for step in range(horizon):
        # advance exog of the last temporal row to the step we are about to predict
        if step > 0 and exog_test_scaled is not None and ts.shape[1] > 1:
            ts[-1, 1:] = exog_test_scaled[step]

        # Overwrite leaky exog columns with values derived from the rolling
        # (own-prediction-augmented) history.  Idempotent at step 0 since
        # y_history still ends at the last observed day there.
        if use_dynamic_exog and ts.shape[1] > 1:
            for col_in_exog, k in lag_col_indices.items():
                raw_lag = float(y_history[-k])
                ts[-1, 1 + col_in_exog] = _scale_lag_value(
                    raw_lag, exog_scaler, col_in_exog
                )
            for col_in_exog, W in rolling_mean_excl_col_indices.items():
                raw_mean = float(np.mean(y_history[-W:]))
                ts[-1, 1 + col_in_exog] = _scale_lag_value(
                    raw_mean, exog_scaler, col_in_exog
                )

        ts_t  = torch.from_numpy(ts).unsqueeze(0).to(device)            # (1, L, F)
        batch = Batch.from_data_list(list(graphs)).to(device)           # B*L = L
        tidx  = batch.ptr[:-1].to(device)
        out   = model(batch, tidx, ts_t)                                # (1, H, 1)
        y_hat = float(out[0, -1, 0].detach().cpu().item())
        preds_scaled.append(y_hat)

        # roll temporal window
        ts = np.vstack([ts[1:], ts[-1:].copy()])
        ts[-1, 0] = y_hat

        # extend the unscaled history with the model's own forecast
        if use_dynamic_exog:
            y_hat_unscaled = float(
                scaler.inverse_transform(np.array([[y_hat]], dtype=np.float32))[0, 0]
            )
            y_history = np.append(y_history, y_hat_unscaled)

        # roll graph deque (push the graph aligned to the day we just predicted)
        if step < len(future_graphs):
            graphs.append(future_graphs[step].clone())
        else:
            graphs.append(graphs[-1].clone())

    preds_scaled = np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()
