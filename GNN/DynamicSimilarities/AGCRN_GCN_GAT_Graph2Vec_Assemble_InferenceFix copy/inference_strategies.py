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

from node_feature_builders import window_node_features  # internal helper
from utils import build_window_ego_graph  # single-source-of-truth ego-graph builder


# ──────────────────────────────────────────────────────────────────────────
# Leak-safe per-step ego-graph REBUILD (GCN/GAT)
# ──────────────────────────────────────────────────────────────────────────
def _strip_graph_aux(data: Data) -> Data:
    """Drop non-tensor debug attributes (``node_order`` / ``node_labels``) so a
    rolling deque can mix graphs produced by different converters.

    ``Batch.from_data_list`` collates the *union* of attribute keys and requires
    every graph to provide each key, so a deque holding both LSTM-built
    (``node_order``) and MLP-built (``node_labels``) ego-graphs would raise
    ``KeyError``.  The forecaster only reads ``x`` / ``edge_index`` /
    ``edge_attr`` / ``ptr``; these label attrs are diagnostics, so stripping
    them is safe (the MLP ``graph_log_out`` path already falls back to integer
    node indices when ``node_labels`` is absent)."""
    for _k in ('node_order', 'node_labels'):
        if _k in data:
            del data[_k]
    return data


def _rebuild_step_graph(ctx: dict, step: int, target_window_vals):
    """Rebuild the GCN/GAT ego-graph for the forecast day ``d`` of ``step``.

    Mirrors the Graph2Vec inference rebuild: the ego-graph for day
    ``d = ctx['first_forecast_col'] + step`` is constructed from the window
    ``[d-W, d-1]`` where the TARGET row is the model's own rolling predictions
    (``target_window_vals``, already in ``df_wide`` scale) and every NEIGHBOUR
    keeps its observed demand.  Topology + edge weighting come from the shared
    ``utils.build_window_ego_graph`` so the rebuilt graph is identical in
    construction to the training graphs (apart from the substituted target row),
    and the node-feature matrix is produced by the same ``nx_to_pyg`` converter
    the dataset used at train time.

    Returns ``(data, neighbour_ids)`` where ``data`` is a PyG ``Data`` ego-graph
    (target node at row 0) and ``neighbour_ids`` is the list of DIRECT neighbour
    item ids selected for this window (``node_order[1:]``).  Returns
    ``(None, [])`` when the window falls outside ``df_wide`` (caller then falls
    back to reusing the previous graph).
    """
    df_wide = ctx['df_wide']
    W = int(ctx['graph_window_size'])
    d = int(ctx['first_forecast_col']) + step
    c0, c1 = d - W, d
    if c0 < 0 or c1 > df_wide.shape[1]:
        return None, []

    date_cols = list(df_wide.columns[c0:c1])
    window_data = df_wide[date_cols]
    tid = ctx['target_id']

    G = build_window_ego_graph(
        window_data, tid, ctx['metric'], ctx['metric_type'], ctx['compute_func'],
        ctx['fixed_threshold'], cat_labels=ctx['cat_labels'],
        enable_edges_within_star=ctx['enable_edges_within_star'],
        enable_second_degree=ctx['enable_second_degree'],
        target_ts_override=list(target_window_vals),
    )

    # node_order = [target] + DIRECT neighbours of the target, exactly as
    # build_pyg_graphs_from_nx_windows does at train time (2nd-degree-only nodes
    # are dropped by the subgraph restriction).
    nbrs = list(G.neighbors(tid)) if tid in G else []
    node_order = [tid] + [n for n in nbrs if n in df_wide.index]

    # window values for those nodes, target row overridden by the predictions.
    wv = df_wide.loc[node_order, date_cols].values.astype(np.float32)
    wv[0] = np.asarray(target_window_vals, dtype=np.float32)

    H = G.subgraph(node_order).copy()
    H.add_nodes_from(node_order)
    g = ctx['nx_to_pyg'](H, node_order, wv, tid,
                         node_feature_mode=ctx['node_feature_mode'])
    return _strip_graph_aux(g), node_order[1:]


def _df_scale_prediction(y_hat_raw: float, target_df_scaler) -> float:
    """Map a raw-units prediction into ``df_wide`` scale (z-score for distance
    metrics; identity for similarity metrics where ``target_df_scaler`` is None)."""
    if target_df_scaler is None:
        return float(y_hat_raw)
    return float(target_df_scaler.transform(
        np.array([[y_hat_raw]], dtype=np.float32))[0, 0])


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
def _recursive_forecast_gcn_lstm_perstep(model, ts_seed, initial_graphs,
                                    future_graphs, exog_test_scaled,
                                    scaler, horizon, device,
                                    target_history_unscaled=None,
                                    lag_col_indices: Optional[Dict[int, int]] = None,
                                    rolling_mean_excl_col_indices: Optional[Dict[int, int]] = None,
                                    exog_scaler=None,
                                    initial_target_window=None,
                                    target_z_scaler=None,
                                    node_feature_mode='stats',
                                    rebuild_ctx=None,
                                    neighbour_log_out: Optional[list] = None):
    """
    One-step-at-a-time inference for the per-step GCN+LSTM.

    Leak-safe graph handling
    ------------------------
    When ``rebuild_ctx`` is provided the ego-graph is **rebuilt from scratch at
    every step** from the window ``[d-W, d-1]`` (target row = the model's own
    rolling predictions, neighbours = observed), exactly like the Graph2Vec
    inference — see ``_rebuild_step_graph``.  This both removes the test-period
    topology leakage of the pre-built ``future_graphs`` and fixes the one-day
    misalignment of the legacy node-feature patch.  When ``rebuild_ctx is None``
    the legacy behaviour is kept: reuse ``future_graphs`` and patch only the
    target node's feature row (back-compat for callers not yet migrated).

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
    neighbour_log_out : optional list.  When provided AND ``rebuild_ctx`` is set,
        one dict ``{'step': int, 'n_neighbours': int, 'neighbours': [ids]}`` is
        appended per forecast step describing the ego-graph rebuilt for the day
        aligned to that step.  Lets the caller plot how the target's neighbourhood
        size evolves across the horizon.

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
    # When rebuilding per step (rebuild_ctx), appended graphs are bare (aux attrs
    # stripped); strip the seed clones too so the deque stays schema-homogeneous
    # and Batch.from_data_list never hits a mixed node_order/node_labels key set.
    if rebuild_ctx is not None:
        for _g in graphs:
            _strip_graph_aux(_g)
    preds_scaled = []

    # Rolling window of target-node values in the same scale as df_wide
    # (raw for similarity metrics, z-scored for distance metrics).  Used both to
    # rebuild the ego-graph each step (rebuild_ctx path) and, in the legacy path,
    # to patch future_graphs[step].x[0] so the GCN never sees the target's actual
    # test-period values during inference.
    target_window = (
        np.asarray(initial_target_window, dtype=np.float32).copy()
        if initial_target_window is not None else None
    )
    if rebuild_ctx is not None and target_window is None:
        raise ValueError(
            "rebuild_ctx requires initial_target_window (the W target values, in "
            "df_wide scale, immediately preceding the first forecast day)."
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
        if rebuild_ctx is not None:
            # Leak-safe rebuild: build the ego-graph for day d = first_forecast_col
            # + step from the CURRENT rolling window [d-W, d-1] (target row = own
            # predictions, neighbours = observed), THEN advance the rolling window
            # with this step's prediction.  Appending AFTER the build keeps the
            # window aligned to the graph's own [d-W, d-1] span (no off-by-one).
            new_graph, _nbr_ids = _rebuild_step_graph(rebuild_ctx, step, target_window)
            if new_graph is None:                      # window out of range → reuse last
                new_graph = graphs[-1].clone()
                _nbr_ids = []
            if neighbour_log_out is not None:
                neighbour_log_out.append({
                    'step': step, 'n_neighbours': len(_nbr_ids),
                    'neighbours': list(_nbr_ids),
                })
            y_node_val = _df_scale_prediction(y_hat_raw, target_z_scaler)
            target_window = np.concatenate([target_window[1:], [y_node_val]])
        else:
            # Legacy path (back-compat): reuse pre-built future_graphs and patch
            # only the target node's feature row.  NOTE: this leaves the edge
            # topology built from the target's real future — prefer rebuild_ctx.
            if step < len(future_graphs):
                new_graph = future_graphs[step].clone()
            else:
                new_graph = graphs[-1].clone()
            if target_window is not None:
                y_node_val = (
                    float(target_z_scaler.transform(np.array([[y_hat_raw]], dtype=np.float32))[0, 0])
                    if target_z_scaler is not None else y_hat_raw
                )
                target_window = np.concatenate([target_window[1:], [y_node_val]])
                new_feats = window_node_features(target_window[None, :], mode=node_feature_mode)
                new_graph.x[0] = torch.from_numpy(new_feats[0])

        graphs.append(new_graph)

    preds_scaled = np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()


def _recursive_forecast_gcn_mlp_perstep(model, ts_seed, initial_graphs,
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
                                    node_feature_mode: str = 'raw',
                                    initial_target_window=None,
                                    target_z_scaler=None,
                                    rebuild_ctx=None,
                                    neighbour_log_out: Optional[list] = None):
    """
    One-step-at-a-time inference for the per-step GCN + MLP.

    Leak-safe graph handling
    ------------------------
    When ``rebuild_ctx`` is provided the ego-graph is rebuilt from scratch every
    step from the window ``[d-W, d-1]`` (target = own predictions, neighbours =
    observed), identically to the LSTM variant and the Graph2Vec inference.  This
    supersedes the legacy ``target_node_std_*`` node-feature patch (which is left
    only for back-compat and is a no-op unless those args are supplied).

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
    neighbour_log_out : optional list.  When provided AND ``rebuild_ctx`` is set,
        one dict ``{'step': int, 'n_neighbours': int, 'neighbours': [ids]}`` is
        appended per forecast step (see the LSTM variant for details).

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
    # When rebuilding per step (rebuild_ctx), appended graphs are bare (aux attrs
    # stripped); strip the seed clones too so the deque stays schema-homogeneous
    # and Batch.from_data_list never hits a mixed node_order/node_labels key set.
    if rebuild_ctx is not None:
        for _g in graphs:
            _strip_graph_aux(_g)
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

    # Rolling target window (df_wide scale) used to rebuild the ego-graph each
    # step when rebuild_ctx is provided (target row = own predictions).
    target_window = (
        np.asarray(initial_target_window, dtype=np.float32).copy()
        if initial_target_window is not None else None
    )
    if rebuild_ctx is not None and target_window is None:
        raise ValueError(
            "rebuild_ctx requires initial_target_window (the W target values, in "
            "df_wide scale, immediately preceding the first forecast day)."
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
        if rebuild_ctx is not None:
            # Leak-safe rebuild: ego-graph for day d = first_forecast_col + step
            # built from the CURRENT rolling window [d-W, d-1] (target row = own
            # predictions, neighbours = observed).  Advance the rolling window
            # AFTER the build so it stays aligned to the graph's [d-W, d-1] span.
            next_graph, _nbr_ids = _rebuild_step_graph(rebuild_ctx, step, target_window)
            if next_graph is None:                     # window out of range → reuse last
                next_graph = graphs[-1].clone()
                _nbr_ids = []
            if neighbour_log_out is not None:
                neighbour_log_out.append({
                    'step': step, 'n_neighbours': len(_nbr_ids),
                    'neighbours': list(_nbr_ids),
                })
            y_hat_raw = float(
                scaler.inverse_transform(np.array([[y_hat]], dtype=np.float32))[0, 0]
            )
            y_node_val = _df_scale_prediction(y_hat_raw, target_z_scaler)
            target_window = np.concatenate([target_window[1:], [y_node_val]])
        else:
            if step < len(future_graphs):
                next_graph = future_graphs[step].clone()
            else:
                next_graph = graphs[-1].clone()

            # ── legacy node-0 patch (no-op unless target_node_std_* supplied) ─
            # Patches only the target feature row; the edge topology still
            # reflects the target's real future.  Prefer rebuild_ctx.
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
