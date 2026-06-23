"""
Graph2Vec integration helpers for the GCN/GAT assemble.

The Graph2Vec variants (``graph2vec_lstm`` / ``graph2vec_mlp``) reuse the whole
per-step pipeline (dataset / collate / training / deque inference) by carrying
each day's Graph2Vec embedding as the single node feature of a 1-node PyG graph
(see ``gcn_models.SimpleGraph2Vec*Forecaster``).

This module provides the two pieces that are NOT shared with the GCN/GAT path:

1. ``embedding_to_single_node_data`` — wrap an embedding vector as a 1-node Data.
2. ``recursive_forecast_graph2vec_perstep`` — recursive inference that, at every
   forecast step, rebuilds the dynamic similarity graph from the model's OWN
   running prediction (never the ground-truth target), re-infers the Graph2Vec
   embedding for that graph, and feeds it forward.  This is the Graph2Vec
   analogue of the GCN per-step inference's node-feature patching: it guarantees
   the target product's future values never leak into its own embedding.
   (Neighbour products' observed values over the horizon are used by design —
   the panel/nowcasting assumption, concern #4.)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Batch, Data

from utils import compute_distances_1vsAll, compute_similarities_1vsAll
from inference_strategies import _scale_lag_value


# ──────────────────────────────────────────────────────────────────────────
# Embedding → PyG single-node graph
# ──────────────────────────────────────────────────────────────────────────
def embedding_to_single_node_data(embedding, target_label=None) -> Data:
    """Wrap a Graph2Vec embedding vector as a 1-node PyG ``Data``.

    The node feature row IS the embedding; the lone self-loop edge is never used
    by the Graph2Vec forecaster (it reads ``x[target_node]`` directly) but keeps
    the object shape-compatible with the GCN dataset / collate / deque.
    """
    x = torch.as_tensor(np.asarray(embedding, dtype=np.float32)).reshape(1, -1)
    data = Data(
        x=x,
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        edge_attr=torch.ones(1, 1, dtype=torch.float32),
        num_nodes=1,
    )
    if target_label is not None:
        data.node_order = [target_label]
    return data


# ──────────────────────────────────────────────────────────────────────────
# Dynamic graph builder (calculated/fixed threshold) — mirrors the Graph2Vec
# pipeline's ``build_dynamic_graph_with_calculated_threshold`` so that the graph
# topology produced at inference matches the one used to fit/produce the
# training embeddings.
# ──────────────────────────────────────────────────────────────────────────
_DISTANCE_METRICS = [
    'euclidean', 'manhattan', 'hamming', 'amplitude_offset', 'slope_consistency',
    'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss',
]


def build_dynamic_graph_with_calculated_threshold(
    target_id, target_preds, df_wide, cat_labels, date_cols, metric,
    fixed_threshold, enable_edges_within_star=True, enable_second_degree=False,
):
    """Build the target's similarity ego-graph for one window.

    ``target_preds`` must already be in the SAME scale as ``df_wide`` (z-scored
    for distance metrics, raw for similarity metrics), so the metric comparison
    against the neighbour rows is consistent with how the training graphs were
    built.
    """
    G = nx.Graph()
    cat = cat_labels.get(target_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
    G.add_node(target_id, cat_label=cat)

    window_data = df_wide[date_cols]
    all_ts = window_data.values
    item_ids = window_data.index.values
    target_ts = np.asarray(target_preds, dtype=float)

    is_distance = metric in _DISTANCE_METRICS
    if is_distance:
        scores = compute_distances_1vsAll(target_ts, all_ts, metric=metric)
    else:
        scores = compute_similarities_1vsAll(target_ts, all_ts, metric=metric)

    active_items_mask = np.sum(np.abs(all_ts), axis=1) > 0
    valid_mask = (item_ids != target_id) & active_items_mask
    if np.sum(np.abs(target_ts)) == 0:
        valid_mask = np.zeros_like(valid_mask, dtype=bool)

    valid_item_ids = item_ids[valid_mask]
    valid_scores = scores[valid_mask]
    valid_original_idxs = np.arange(len(item_ids))[valid_mask]

    if is_distance:
        final_mask = valid_scores <= fixed_threshold
    else:
        final_mask = valid_scores >= fixed_threshold

    selected_scores = valid_scores[final_mask]
    selected_ids = valid_item_ids[final_mask]
    selected_orig_idxs = valid_original_idxs[final_mask]

    neighbor_indices = []
    for orig_idx, score_val, other_id in zip(selected_orig_idxs, selected_scores, selected_ids):
        neighbor_indices.append((orig_idx, other_id))
        if not G.has_node(other_id):
            cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
            G.add_node(other_id, cat_label=cat_other)
        G.add_edge(target_id, other_id, weight=float(score_val))

    if enable_edges_within_star and len(neighbor_indices) > 1:
        n_ts_array = np.array([all_ts[idx] for idx, _ in neighbor_indices])
        n_ids = [oid for _, oid in neighbor_indices]
        for i, n_id1 in enumerate(n_ids):
            n1_ts = n_ts_array[i]
            if is_distance:
                n_scores = compute_distances_1vsAll(n1_ts, n_ts_array, metric=metric)
            else:
                n_scores = compute_similarities_1vsAll(n1_ts, n_ts_array, metric=metric)
            for j in range(i + 1, len(n_ids)):
                condition = (n_scores[j] <= fixed_threshold) if is_distance else (n_scores[j] >= fixed_threshold)
                if condition:
                    if np.sum(np.abs(n1_ts)) > 0 and np.sum(np.abs(n_ts_array[j])) > 0:
                        G.add_edge(n_id1, n_ids[j], weight=float(n_scores[j]))

    if enable_second_degree and len(neighbor_indices) > 0:
        for orig_idx, _ in neighbor_indices:
            target_neighbor_ts = all_ts[orig_idx]
            if is_distance:
                n_scores = compute_distances_1vsAll(target_neighbor_ts, all_ts, metric=metric)
            else:
                n_scores = compute_similarities_1vsAll(target_neighbor_ts, all_ts, metric=metric)
            for valid_idx, is_valid in enumerate(valid_mask):
                if is_valid and valid_idx != orig_idx:
                    val_sub = n_scores[valid_idx]
                    other_id = item_ids[valid_idx]
                    add_edge = (val_sub <= fixed_threshold) if is_distance else (val_sub >= fixed_threshold)
                    if add_edge:
                        if not G.has_node(other_id):
                            cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
                            G.add_node(other_id, cat_label=cat_other)
                        if not G.has_edge(item_ids[orig_idx], other_id):
                            G.add_edge(item_ids[orig_idx], other_id, weight=float(val_sub))

    return G


# ──────────────────────────────────────────────────────────────────────────
# Recursive (one-step-at-a-time) inference for the Graph2Vec per-step variants
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def recursive_forecast_graph2vec_perstep(
    model, ts_seed, initial_graphs, exog_test_scaled, scaler, horizon, device,
    # ── dynamic Graph2Vec rebuild ──
    graph2vec_model, df_wide, cat_labels, target_id, metric, fixed_threshold,
    graph_window_size, first_forecast_col, target_seed_window_dfscale,
    enable_edges_within_star=False, enable_second_degree=False,
    target_df_scaler=None,
    # ── leak-safe dynamic exog (same contract as the GCN inference) ──
    target_history_unscaled=None,
    lag_col_indices: Optional[Dict[int, int]] = None,
    rolling_mean_excl_col_indices: Optional[Dict[int, int]] = None,
    exog_scaler=None,
    neighbour_log_out: Optional[list] = None,
):
    """One-step-at-a-time inference for the per-step Graph2Vec LSTM/MLP.

    Mirrors ``inference_strategies._recursive_forecast_gcn_lstm_perstep`` (same
    LSTM-seed / exog-rolling / leak-safe-exog logic) but replaces the rolling
    graph deque's "patch node-0 features" step with a full dynamic-graph rebuild
    + Graph2Vec re-inference:

    At each step we predict day ``d = first_forecast_col + step`` from the deque,
    then build the graph aligned to ``d`` (days ``d-W .. d-1`` — strictly before
    ``d``, matching ``_align_pyg_windows_to_timeline``) using the model's OWN
    rolling target window (in ``df_wide`` scale), infer its Graph2Vec embedding,
    and push it.  Initial deque graphs come from the observed validation period.

    Parameters mirror the GCN inference; the Graph2Vec-specific ones are:

    graph2vec_model           : fitted CustomGraph2Vec (exposes ``infer``)
    df_wide                   : item × date pivot in the metric's scale
                                (z-scored for distance, raw for similarity)
    target_seed_window_dfscale: (W,) target values in df_wide scale for the W
                                days immediately before the test period
    first_forecast_col        : integer column index in ``df_wide`` of the first
                                forecast day (= product_offset + test_start_idx)
    target_df_scaler          : product StandardScaler used to map a raw
                                prediction into df_wide scale (distance metrics);
                                None → raw (similarity metrics).
    neighbour_log_out         : optional list; one dict
                                ``{'step', 'n_neighbours', 'neighbours'}`` is
                                appended per forecast step describing the dynamic
                                similarity graph rebuilt for that day (lets the
                                caller plot neighbourhood-size evolution).
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
            raise ValueError("exog_scaler is required when lag/rolling col indices are provided.")
        y_history = np.asarray(target_history_unscaled, dtype=np.float32).copy()

    graphs: "deque[Data]" = deque((g.clone() for g in initial_graphs), maxlen=L)
    preds_scaled = []

    W = int(graph_window_size)
    # Rolling target window in df_wide scale (ends at day first_forecast_col-1).
    target_window = deque(
        np.asarray(target_seed_window_dfscale, dtype=np.float32).tolist(),
        maxlen=W,
    )

    for step in range(horizon):
        # advance exog of the last LSTM row to the step we are about to predict
        if step > 0 and exog_test_scaled is not None and ts.shape[1] > 1:
            ts[-1, 1:] = exog_test_scaled[step]

        # overwrite leaky exog columns from the (own-prediction-augmented) history
        if use_dynamic_exog and ts.shape[1] > 1:
            for col_in_exog, k in lag_col_indices.items():
                ts[-1, 1 + col_in_exog] = _scale_lag_value(float(y_history[-k]), exog_scaler, col_in_exog)
            for col_in_exog, Wr in rolling_mean_excl_col_indices.items():
                ts[-1, 1 + col_in_exog] = _scale_lag_value(float(np.mean(y_history[-Wr:])), exog_scaler, col_in_exog)

        ts_t  = torch.from_numpy(ts).unsqueeze(0).to(device)            # (1, L, F)
        batch = Batch.from_data_list(list(graphs)).to(device)           # B*L = L
        tidx  = batch.ptr[:-1].to(device)
        out   = model(batch, tidx, ts_t)                                # (1, H, 1)
        y_hat = float(out[0, -1, 0].detach().cpu().item())
        preds_scaled.append(y_hat)

        # roll LSTM window
        ts = np.vstack([ts[1:], ts[-1:].copy()])
        ts[-1, 0] = y_hat

        # inverse-scale prediction (raw target units)
        y_hat_raw = float(scaler.inverse_transform(np.array([[y_hat]], dtype=np.float32))[0, 0])
        if use_dynamic_exog:
            y_history = np.append(y_history, y_hat_raw)

        # ── build the graph aligned to day d = first_forecast_col + step ──────
        # aligned[d] uses days [d-W, d-1] (strictly before d) → the CURRENT
        # rolling window (which ends at d-1, before appending this prediction).
        d = first_forecast_col + step
        c0, c1 = d - W, d
        if c0 >= 0 and c1 <= df_wide.shape[1] and len(target_window) == W:
            date_cols = list(df_wide.columns[c0:c1])
            G = build_dynamic_graph_with_calculated_threshold(
                target_id=target_id,
                target_preds=list(target_window),
                df_wide=df_wide,
                cat_labels=cat_labels,
                date_cols=date_cols,
                metric=metric,
                fixed_threshold=fixed_threshold,
                enable_edges_within_star=enable_edges_within_star,
                enable_second_degree=enable_second_degree,
            )
            new_emb = graph2vec_model.infer([G])[0]
            new_graph = embedding_to_single_node_data(new_emb, target_label=target_id)
            if neighbour_log_out is not None:
                _nbr_ids = list(G.neighbors(target_id)) if target_id in G else []
                neighbour_log_out.append({
                    'step': step, 'n_neighbours': len(_nbr_ids),
                    'neighbours': _nbr_ids,
                })
        else:
            # Out-of-range window (shouldn't happen for valid splits): reuse last.
            new_graph = graphs[-1].clone()
            if neighbour_log_out is not None:
                neighbour_log_out.append({
                    'step': step, 'n_neighbours': 0, 'neighbours': [],
                })

        # advance the rolling target window with this step's prediction
        # (df_wide scale) for the NEXT day's graph.
        y_node_val = (
            float(target_df_scaler.transform(np.array([[y_hat_raw]], dtype=np.float32))[0, 0])
            if target_df_scaler is not None else y_hat_raw
        )
        target_window.append(y_node_val)

        graphs.append(new_graph)

    preds_scaled = np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()
