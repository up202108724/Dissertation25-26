"""
Recursive (one-step-at-a-time) inference for the PER-STEP GCN + MLP forecaster.

Direct port of ``LSTM_GCN_1_graph_per_lookback/inference.py``.  Since the
``SimpleGCNMLPForecaster`` exposes the same forward signature as the LSTM
sibling — ``model(pyg_batch, target_node_indices, ts_seq) -> (B, H, 1)`` —
this rollout logic is model-agnostic and works as-is for the MLP head.

At each forecasting step we:

  1. Run the model on the current (L-graph deque, L-row ts_seq).
  2. Roll the lookback window forward (drop oldest target value, append ŷ;
     advance the exogenous row to the step we are about to predict).
  3. Roll the graph deque forward: drop the oldest graph, append the new
     "current" ego-graph for the day just predicted (taken from
     ``future_graphs`` when provided, otherwise reuse the last graph).

Also exposes a small ``_align_pyg_windows_to_timeline`` helper used by the
runner to turn the per-window PyG list into a per-day list.
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
    *strictly before* day t (days t-W .. t-1).  Matches the
    GCNTimeSeriesDataset convention and guarantees no label leakage (the
    graph chosen for the last lookback step never includes the label day).
    """
    if step_size != 1:
        raise NotImplementedError("Per-day alignment helper currently assumes step_size=1")

    pad = _make_pad_graph(pyg_windows[0])
    aligned = [pad] * window_size + list(pyg_windows)
    if len(aligned) < T:
        aligned += [aligned[-1]] * (T - len(aligned))
    else:
        aligned = aligned[:T]
    return aligned


# ──────────────────────────────────────────────────────────────────────────
# Helpers for leakage-safe dynamic exog (lags, rolling means)
# ──────────────────────────────────────────────────────────────────────────
def _scale_lag_value(raw_value: float, exog_scaler, col_idx: int,
                     exog_col_name: str = None) -> float:
    """
    Apply a single-column scaler transform on one scalar.  Works for both
    plain sklearn scalers and ExogenousScaler (type-aware pass-through).
    """
    if exog_scaler is None:
        return float(raw_value)
    # ExogenousScaler — use the per-column scaler directly if available.
    if hasattr(exog_scaler, "scalers"):
        if exog_col_name is not None and exog_col_name in exog_scaler.scalers:
            col_scaler = exog_scaler.scalers[exog_col_name]
            return float(col_scaler.transform([[raw_value]])[0, 0])
        # Binary / cyclical column: pass-through
        return float(raw_value)
    # Plain sklearn scaler (MinMaxScaler, StandardScaler, etc.)
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
# Leakage-safe target-node feature patching
# ──────────────────────────────────────────────────────────────────────────
def _patch_target_node_features(graph: Data, std_buffer, node_feature_mode: str) -> None:
    """
    Overwrite row 0 (the target node) of ``graph.x`` with features derived
    from ``std_buffer`` — the rolling window of std-scaled *predicted* target
    values — so the GCN never reads ground-truth test values for the series
    being forecast.  Mutates ``graph`` in place (caller passes a clone).

    ``std_buffer`` holds ``window_size`` std-scaled values aligned to the
    graph's node-feature window ``[day - window_size, day)``.  Feature layout
    must match ``gcn_tdmlpdataset``: 'raw' = the sequence itself, 'stats' =
    ``[mean, std, min, max, first, last, slope, sum]``, 'catch22'/'catch24' =
    the catch22 shape/dynamics descriptor.
    """
    w = np.asarray(list(std_buffer), dtype=np.float32)
    if node_feature_mode == 'raw':
        if graph.x.shape[1] != w.shape[0]:
            return  # width mismatch (e.g. degenerate pad graph) — skip
        graph.x[0, :] = torch.from_numpy(w).to(graph.x.dtype)
    elif node_feature_mode == 'stats':
        feats = np.array([
            w.mean(), w.std(), w.min(), w.max(),
            w[0], w[-1],
            (w[-1] - w[0]) / max(w.shape[0] - 1, 1),
            w.sum(),
        ], dtype=np.float32)
        if graph.x.shape[1] != feats.shape[0]:
            return
        graph.x[0, :] = torch.from_numpy(feats).to(graph.x.dtype)
    elif node_feature_mode in ('catch22', 'catch24'):
        from gcn_tdmlpdataset import _window_node_features_catch22
        feats = _window_node_features_catch22(
            w[None, :], catch24=(node_feature_mode == 'catch24'))[0]
        if graph.x.shape[1] != feats.shape[0]:
            return
        graph.x[0, :] = torch.from_numpy(feats).to(graph.x.dtype)
    else:
        raise ValueError(
            "node_feature_mode must be 'raw', 'stats', 'catch22' or 'catch24', "
            f"got {node_feature_mode!r}"
        )


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
                                    exog_scaler=None,
                                    exog_cols=None,
                                    graph_log_out: Optional[list] = None,
                                    step_callback=None,
                                    target_node_std_scaler=None,
                                    target_node_std_seed=None,
                                    node_feature_mode: str = 'raw'):
    """
    One-step-at-a-time inference for the per-step GCN + MLP.

    Parameters
    ----------
    model            : SimpleGCNMLPForecaster (per-step variant)
    ts_seed          : (L, 1+n_exog) scaled lookback seed; row L-1 already
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
        step.  Typically ``np.concatenate([train_raw, val_raw])``.  Extended
        at each step with the model's own (inverse-scaled) prediction so
        subsequent lookups never peek at ground-truth test values.
    lag_col_indices : ``{col_in_exog -> k}`` (e.g. ``{7: 1, 8: 7, 9: 30}``
        if positions 7/8/9 in the scaled exog vector are lag_1 / lag_7 /
        lag_30).  Overwrites each step's last-row exog with
        ``scale(target_history_unscaled[-k])``.
    rolling_mean_excl_col_indices : ``{col_in_exog -> W}`` for
        ``rolling_mean_excl_W`` features (mean of the W RAW target values
        strictly preceding the day being predicted, i.e.
        ``mean(target_history_unscaled[-W:])``).
    exog_scaler     : sklearn scaler used to scale the exog matrix.  Needed
        whenever any of the col-index dicts is provided.
    target_node_std_scaler : sklearn StandardScaler fit on the target product
        (``product_scalers[product_id]``).  When provided together with
        ``target_node_std_seed``, node-0 features in every future graph are
        replaced by a prediction-derived window instead of the leaked
        ground-truth test values.
    target_node_std_seed   : 1-D np.ndarray of length ``window_size`` with
        the std-scaled target values for the ``window_size`` days immediately
        before the test period (from ``df_wide_scaled``).  Initialises the
        rolling feature buffer.
    node_feature_mode : 'raw' or 'stats' — must match ``NODE_FEATURE_MODE``.

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

    # ── Rolling std-scaled buffer for target-node feature patching ────────
    # Replaces real test-period values in future_graphs' node-0 features with
    # the model's own rolling predictions, eliminating label leakage.
    _patch_target = (
        target_node_std_scaler is not None
        and target_node_std_seed is not None
    )
    if _patch_target:
        _window_size = len(target_node_std_seed)
        _std_buffer: deque = deque(
            np.asarray(target_node_std_seed, dtype=np.float32).tolist(),
            maxlen=_window_size,
        )

    for step in range(horizon):
        # advance exog of the last lookback row to the step we are about to predict
        if step > 0 and exog_test_scaled is not None and ts.shape[1] > 1:
            ts[-1, 1:] = exog_test_scaled[step]

        # Overwrite leaky exog columns with values derived from the rolling
        # (own-prediction-augmented) history.  Idempotent at step 0 since
        # y_history still ends at the last observed day there.
        if use_dynamic_exog and ts.shape[1] > 1:
            for col_in_exog, k in lag_col_indices.items():
                raw_lag = float(y_history[-k])
                col_name = exog_cols[col_in_exog] if exog_cols else None
                ts[-1, 1 + col_in_exog] = _scale_lag_value(
                    raw_lag, exog_scaler, col_in_exog, col_name
                )
            for col_in_exog, W in rolling_mean_excl_col_indices.items():
                raw_mean = float(np.mean(y_history[-W:]))
                col_name = exog_cols[col_in_exog] if exog_cols else None
                ts[-1, 1 + col_in_exog] = _scale_lag_value(
                    raw_mean, exog_scaler, col_in_exog, col_name
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

        # roll lookback window: shift left, append a new last row carrying ŷ
        ts = np.vstack([ts[1:], ts[-1:].copy()])
        ts[-1, 0] = y_hat

        # extend the unscaled history with the model's own forecast so that
        # the next step's lag_k / rolling_mean_excl_W reads ŷ instead of
        # the ground-truth value.
        if use_dynamic_exog:
            y_hat_unscaled = float(
                scaler.inverse_transform(np.array([[y_hat]], dtype=np.float32))[0, 0]
            )
            y_history = np.append(y_history, y_hat_unscaled)

        # roll graph deque (push the graph aligned to the day we just predicted)
        if step < len(future_graphs):
            next_graph = future_graphs[step].clone()
        else:
            next_graph = graphs[-1].clone()

        
        # ── leakage fix: patch the target node (row 0) of next_graph ──────
        # next_graph is aligned to global day (test_start + step); its node-0
        # feature window is [day - window_size, day), which for step >= 1
        # contains GROUND-TRUTH test values of the series we forecast.
        # Replace that row with the model's own rolling predictions
        # (std-scaled with the product's StandardScaler) so the GCN never
        # reads the future.  _std_buffer currently spans exactly this graph's
        # node-feature window; advance it with ŷ_step afterwards.
        if _patch_target:
            _patch_target_node_features(next_graph, _std_buffer, node_feature_mode)
            _y_hat_unscaled_g = float(
                scaler.inverse_transform(np.array([[y_hat]], dtype=np.float32))[0, 0]
            )
            _z_val = float(
                target_node_std_scaler.transform(
                    np.array([[_y_hat_unscaled_g]], dtype=np.float32)
                )[0, 0]
            )
            _std_buffer.append(_z_val)

        graphs.append(next_graph)

    preds_scaled = np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()
