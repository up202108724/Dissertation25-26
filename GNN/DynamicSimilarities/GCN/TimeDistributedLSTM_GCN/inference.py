"""
Recursive (one-step-at-a-time) inference for the PER-STEP GCN + LSTM forecaster.

The per-step model expects *L* graphs per sample (one per lookback day), so at
inference we maintain a rolling deque of length L of ego-graphs alongside the
LSTM lookback window.  At each forecasting step we:

  1. Run the model on the current (L-graph deque, L-row ts_seq).
  2. Roll the LSTM window forward (drop oldest target value, append ŷ;
     advance the exogenous row to the step we are about to predict).
  3. Roll the graph deque forward: drop the oldest graph, append a new
     "current" graph.  If the caller supplies a sequence of *future*
     ego-graphs via ``future_graphs`` we use those (one per recursive
     step); otherwise we reuse the last graph and only refresh its
     target-node feature row from the rolling target-window values.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from gcn_lstm_dataset import (  # internal helpers
    _window_node_features,
    _window_node_features_raw_values,
)


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
    *strictly before* day t (days t-W .. t-1).  This matches the
    GCNTimeSeriesDataset / Graph2Vec convention and is the pad that
    GUARANTEES no label leakage in the per-step variant (the graph chosen
    for the last lookback step never includes the label day).
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
# Recursive inference (PER-STEP: deque of L graphs)
# ──────────────────────────────────────────────────────────────────────────
def _scale_lag_value(raw_value: float, exog_scaler, col_idx: int) -> float:
    """
    Apply a single-column MinMaxScaler / StandardScaler transform on one scalar.
    Works for any sklearn scaler that exposes ``transform`` on a 2-D array — we
    just embed the scalar into a zero row, transform, and pick the column back.
    """
    if exog_scaler is None:
        return float(raw_value)
    n_features = getattr(exog_scaler, "n_features_in_", None)
    if n_features is None:
        # Fallback: infer from data_min_ (MinMaxScaler) or mean_ (StandardScaler)
        ref = getattr(exog_scaler, "data_min_", None)
        if ref is None:
            ref = getattr(exog_scaler, "mean_", None)
        if ref is None:
            raise ValueError("exog_scaler must expose n_features_in_, data_min_ or mean_")
        n_features = len(ref)
    row = np.zeros((1, n_features), dtype=np.float32)
    row[0, col_idx] = float(raw_value)
    return float(exog_scaler.transform(row)[0, col_idx])


@torch.no_grad()
def _recursive_forecast_gcn_perstep(model, ts_seed, initial_graphs,
                                    future_graphs, exog_test_scaled,
                                    scaler, horizon, device,
                                    target_history_unscaled=None,
                                    lag_col_indices: Optional[Dict[int, int]] = None,
                                    rolling_mean_excl_col_indices: Optional[Dict[int, int]] = None,
                                    exog_scaler=None,
                                    initial_target_window=None,
                                    target_z_scaler=None,
                                    node_feature_mode='stats',
                                    graph_log_out: Optional[list] = None,
                                    step_callback=None):
    """
    One-step-at-a-time inference for the per-step GCN+LSTM.

    Parameters
    ----------
    model            : SimpleGCNLSTMForecaster (per-step variant)
    ts_seed          : (L, 1+n_exog) scaled LSTM seed; row L-1 already
                       contains the exog for the FIRST predicted step.
    initial_graphs   : list of L Data ego-graphs (oldest..newest) aligned to
                       the L lookback days at inference start.
    future_graphs    : list of length ``horizon`` of ego-graphs aligned to
                       each successive forecast day.  Each step appends one
                       and drops the oldest from the rolling deque.
    exog_test_scaled : (horizon, n_exog) scaled exog for the test window.
    scaler           : sklearn MinMaxScaler fit on the target.
    horizon          : number of recursive steps.
    device           : torch.device

    Leakage-safe dynamic exog (optional — pass ``target_history_unscaled``
    + ``exog_scaler`` and at least one of the col-index dicts to activate):
    target_history_unscaled : 1-D np.ndarray with the RAW (unscaled) target
        history that ends at the day immediately BEFORE the first forecast
        step.  Typically ``np.concatenate([train_raw, val_raw])``.  It is
        extended at each step with the model's own (inverse-scaled)
        prediction so that subsequent lookups never peek at ground-truth
        test values.
    lag_col_indices : mapping ``{col_in_exog -> k}`` (e.g. ``{7: 1, 8: 7, 9: 30}``
        if positions 7/8/9 in the scaled exog vector are lag_1/lag_7/lag_30).
        For every step and every (col, k) in the dict the row injected into
        the LSTM is overwritten with ``scale(target_history_unscaled[-k])``.
    rolling_mean_excl_col_indices : mapping ``{col_in_exog -> W}`` for
        ``rolling_mean_excl_W`` features (mean of the W RAW target values
        strictly preceding the day being predicted, i.e.
        ``mean(target_history_unscaled[-W:])`` since the history already
        excludes the current step).
    exog_scaler     : the sklearn scaler used to scale the exog matrix.  Needed
        whenever any of the col-index dicts is provided.
    graph_log_out   : optional list; when provided, one dict per edge of the
        most-recent graph used at each step is appended (step, n_nodes,
        src_node, tgt_node, edge_weight) for offline neighbourhood logging.
    step_callback   : optional callable ``step_callback(step_idx)`` invoked
        once per forecast step (e.g. to save the inferred ego-graph plot).

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

    # Rolling window of target-node values in the same scale as df_wide
    # (raw for similarity metrics, z-scored for distance metrics).
    # Patched into future_graphs[step].x[0] after each step so that actual
    # test-period values are never seen by the GCN during inference.
    target_window = (
        np.asarray(initial_target_window, dtype=np.float32).copy()
        if initial_target_window is not None else None
    )

    for step in range(horizon):
        # advance exog of the last LSTM row to the step we are about to predict
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
                # rolling_mean_excl_W[t] = mean(value[t-W : t])
                # y_history ends at day t-1 -> last W values are exactly the
                # "excl" window for the day we are about to predict.
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

        # ── record neighbourhood / adjacency for this step ────────────────
        if graph_log_out is not None:
            g = graphs[-1]  # most recent graph used in this prediction
            ei = g.edge_index.cpu().numpy()           # (2, n_edges)
            ea = g.edge_attr.cpu().numpy().flatten()  # (n_edges,)
            labels = getattr(g, 'node_labels', None)
            n_nodes = int(g.num_nodes)
            for eidx in range(ei.shape[1]):
                s_idx, t_idx = int(ei[0, eidx]), int(ei[1, eidx])
                src_label = labels[s_idx] if labels is not None else s_idx
                tgt_label = labels[t_idx] if labels is not None else t_idx
                graph_log_out.append({
                    'step': step,
                    'n_nodes': n_nodes,
                    'src_node': src_label,
                    'tgt_node': tgt_label,
                    'edge_weight': float(ea[eidx]),
                })

        if step_callback is not None:
            step_callback(step)

        # roll LSTM window
        ts = np.vstack([ts[1:], ts[-1:].copy()])
        ts[-1, 0] = y_hat

        # Inverse-scale prediction — reused by both y_history and the node-feature update.
        y_hat_raw = float(scaler.inverse_transform(np.array([[y_hat]], dtype=np.float32))[0, 0])

        # extend the unscaled history with the model's own forecast so that
        # the next step's lag_k / rolling_mean_excl_W reads ŷ instead of
        # the ground-truth value.
        if use_dynamic_exog:
            y_history = np.append(y_history, y_hat_raw)

        # roll graph deque (push the graph aligned to the day we just predicted)
        if step < len(future_graphs):
            new_graph = future_graphs[step].clone()
        else:
            new_graph = graphs[-1].clone()

        # Patch target-node (row 0) features with the model's own rolling predictions,
        # removing the test-period leakage baked into pre-built future_graphs for steps >= 1.
        if target_window is not None:
            y_node_val = (
                float(target_z_scaler.transform(np.array([[y_hat_raw]], dtype=np.float32))[0, 0])
                if target_z_scaler is not None else y_hat_raw
            )
            target_window = np.concatenate([target_window[1:], [y_node_val]])
            if node_feature_mode == 'raw':
                new_feats = _window_node_features_raw_values(target_window[None, :])
            else:
                new_feats = _window_node_features(target_window[None, :])
            new_graph.x[0] = torch.from_numpy(new_feats[0])

        graphs.append(new_graph)

    preds_scaled = np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()


