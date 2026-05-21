import torch
import numpy as np
from typing import Optional
from utils import compute_distances_1vsAll, compute_similarities_1vsAll
import networkx as nx
import pandas as pd
def build_dynamic_graph_with_calculated_threshold(target_id, target_preds, df_wide, cat_labels, date_cols, metric, fixed_threshold, enable_edges_within_star=True, enable_second_degree=False, node_features: list = None):
    from torch_geometric.data import Data
    try:
        from gnn_lstm_pyg import generate_node_features
    except ImportError:
        from gnn_pyg import generate_node_features

    # Extract data for the window
    window_data = df_wide[date_cols]
    all_ts = window_data.values
    item_ids = window_data.index.values
    
    target_ts = np.array(target_preds)
    distance_metrics=['euclidean','manhattan', 'hamming', 'amplitude_offset', 'slope_consistency', 'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
    similarity_metrics=['pearson', 'spearman', 'kendall']
   
    metric_type = 'distance' if metric in distance_metrics else 'similarity'

    # Compute metric-specific scores
    if metric_type == 'distance':
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
    
    if len(valid_scores) == 0:
        mask = np.array([], dtype=bool)
    else:
        if metric_type == 'distance':
            mask = valid_scores <= fixed_threshold
        else:
            mask = valid_scores >= fixed_threshold
        
    selected_scores = valid_scores[mask]
    selected_ids = valid_item_ids[mask]
    selected_orig_idxs = valid_original_idxs[mask]
    
    # Garantir ordem determinística
    sort_idx = np.argsort(selected_ids)
    selected_scores = selected_scores[sort_idx]
    selected_ids = selected_ids[sort_idx]
    selected_orig_idxs = selected_orig_idxs[sort_idx]

    graph_nodes = [target_id] + selected_ids.tolist()
    node_to_idx = {n_id: idx for idx, n_id in enumerate(graph_nodes)}
    
    edge_list = []
    edge_weights = []

    neighbor_indices = selected_orig_idxs.tolist()
    
    for val, other_id in zip(selected_scores, selected_ids):
        u = 0
        v = node_to_idx[other_id]
        
        edge_list.append([u, v])
        edge_list.append([v, u])
        
        weight = float(val)
        if metric_type == 'distance':
            weight = 1.0 / (1.0 + weight)
        else:
            weight = max(0.0, weight)
            
        edge_weights.extend([weight, weight])
            
    if enable_edges_within_star and len(neighbor_indices) > 1:
        all_neighbors_ts = all_ts[neighbor_indices]
        for i, idx1 in enumerate(neighbor_indices):
            target_neighbor_ts = all_ts[idx1]
            if metric_type == 'distance':
                n_scores = compute_distances_1vsAll(target_neighbor_ts, all_neighbors_ts, metric=metric)
            else:
                n_scores = compute_similarities_1vsAll(target_neighbor_ts, all_neighbors_ts, metric=metric)
                
            for j, (idx2, val_sub) in enumerate(zip(neighbor_indices, n_scores)):
                if i < j:
                    if metric_type == 'distance':
                        if val_sub <= fixed_threshold:
                            edge_weight = 1.0 / (1.0 + float(val_sub))
                            u, v = node_to_idx[item_ids[idx1]], node_to_idx[item_ids[idx2]]
                            edge_list.append([u, v])
                            edge_list.append([v, u])
                            edge_weights.extend([edge_weight, edge_weight])
                    else:
                        if val_sub >= fixed_threshold:
                            edge_weight = max(0.0, float(val_sub))
                            u, v = node_to_idx[item_ids[idx1]], node_to_idx[item_ids[idx2]]
                            edge_list.append([u, v])
                            edge_list.append([v, u])
                            edge_weights.extend([edge_weight, edge_weight])
                        
    if enable_second_degree and len(neighbor_indices) > 0:
        for idx1 in neighbor_indices:
            target_neighbor_ts = all_ts[idx1]
            if metric_type == 'distance':
                n_scores = compute_distances_1vsAll(target_neighbor_ts, all_ts, metric=metric)
            else:
                n_scores = compute_similarities_1vsAll(target_neighbor_ts, all_ts, metric=metric)
                
            for valid_idx, is_valid in enumerate(valid_mask):
                if is_valid and valid_idx != idx1:  
                    val_sub = n_scores[valid_idx]
                    other_id = item_ids[valid_idx]
                    
                    add_edge = False
                    if metric_type == 'distance':
                        if val_sub <= fixed_threshold:
                            add_edge = True
                            edge_weight = 1.0 / (1.0 + float(val_sub))
                    else:
                        if val_sub >= fixed_threshold:
                            add_edge = True
                            edge_weight = max(0.0, float(val_sub))
                            
                    if add_edge:
                        if other_id not in node_to_idx:
                            node_to_idx[other_id] = len(graph_nodes)
                            graph_nodes.append(other_id)
                        
                        u, v = node_to_idx[item_ids[idx1]], node_to_idx[other_id]
                        edge_list.append([u, v])
                        edge_list.append([v, u])
                        edge_weights.extend([edge_weight, edge_weight])
                            
    # Construct PyTorch Tensors
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    # Build feature matrix X — always store raw ts first so downstream code can
    # reliably extract neighbor ts via orig_feat[:graph_window_size].
    _feats = node_features if node_features is not None else [
        'last_demand', 'mean7', 'mean_all', 'std_all', 'zero_ratio', 'slope', 'min_v', 'max_v'
    ]
    _storage_feats = ['ts'] + [f for f in _feats if f != 'ts']
    x_matrix = []
    for n_id in graph_nodes:
        if n_id == target_id:
            features = generate_node_features(target_ts, selected_features=_storage_feats)
        else:
            row = window_data.loc[n_id].values
            features = generate_node_features(row, selected_features=_storage_feats)
        x_matrix.append(features)
        
    x = torch.tensor(np.array(x_matrix), dtype=torch.float)

    # PyG Data Object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.central_node_idx = 0
    data.target_id = target_id
    
    return data

# ---------------------------------------------------------------------------
# GNN+LSTM recursive inference  (SimpleGNNLSTMForecaster)
# ---------------------------------------------------------------------------

def recursive_inference_pure_sage(
    model: torch.nn.Module,
    scaler,
    recent_history: np.ndarray,
    future_exog: np.ndarray,
    target_channel: int = 0,
    device: Optional[str] = None,
    # --- Graph parameters ---
    df_wide: pd.DataFrame = None,
    cat_labels: dict = None,
    target_id: int = None,
    metric: str = 'spearman',
    fixed_threshold: float = 0.5,
    enable_edges_within_star: bool = False,
    enable_second_degree: bool = False,
    past_dates: list = None,
    future_dates: list = None,
    graph_window_size: int = 15,
    include_cal_lookback: bool = False,
    node_features=None,
    cal_columns=None,
) -> np.ndarray:
    """
    Recursive 1-step-ahead inference for SimpleGNNLSTMForecaster.

    Mirrors GraphSAGE_LSTM/graphsageinference.py exactly:
      • ONE graph per step (window ending at the last observed timestep).
      • Target-node features rebuilt with generate_node_features.
      • ts_seq[t] = (value_t, cal_{t+1}) fed to LSTM.
    """
    from torch_geometric.data import Batch
    from gnn_lstm_pyg import generate_node_features

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    C_in         = recent_history.shape[1]
    lookback     = recent_history.shape[0]
    exog_indices = [i for i in range(C_in) if i != target_channel]
    cal_dim      = len(exog_indices)
    horizon      = len(future_exog)

    x_scaled = recent_history.copy()
    x_scaled[:, target_channel:target_channel+1] = scaler.transform(
        recent_history[:, target_channel:target_channel+1]
    )

    target_window_scaled = x_scaled[:, target_channel].copy()   # (lookback,)
    rolling_cal = x_scaled[:, exog_indices].copy() if cal_dim > 0 else None  # (lookback, cal_dim)
    cal_window  = x_scaled[:, exog_indices].copy() if (include_cal_lookback and cal_dim > 0) else None

    all_dates = list(past_dates) + list(future_dates)

    _dummy_ts  = np.zeros(lookback, dtype=np.float32)
    _dummy_cal = np.zeros(cal_dim,  dtype=np.float32) if cal_dim > 0 else None
    _dummy_lb  = (np.zeros((lookback, cal_dim), dtype=np.float32)
                  if include_cal_lookback and cal_dim > 0 else None)
    feature_dim = len(generate_node_features(
        _dummy_ts, cal_next=_dummy_cal, cal_lookback=_dummy_lb,
        selected_features=node_features, cal_columns=cal_columns,
    ))

    preds_unscaled = []
    model = model.to(device).eval()

    with torch.no_grad():
        for i in range(horizon):
            # ── 1. Build one ego-graph for the current window ──────────────
            t_idx_global = len(past_dates) - 1 + i
            date_start   = max(0, t_idx_global - graph_window_size + 1)
            window_dates = all_dates[date_start : t_idx_global + 1]

            target_preds_for_graph = (
                df_wide.loc[target_id, window_dates].values.astype(float).copy()
            )
            for f_idx, f_date in enumerate(future_dates[:i]):
                if f_date in window_dates:
                    w_idx = list(window_dates).index(f_date)
                    target_preds_for_graph[w_idx] = preds_unscaled[f_idx]

            G_data = build_dynamic_graph_with_calculated_threshold(
                target_id=target_id, target_preds=target_preds_for_graph,
                df_wide=df_wide, cat_labels=cat_labels, date_cols=window_dates,
                metric=metric, fixed_threshold=fixed_threshold,
                enable_edges_within_star=enable_edges_within_star,
                enable_second_degree=enable_second_degree,
                node_features=node_features,
            )

            # ── 2. Override node features ────────────────────────────────
            n_nodes = G_data.x.shape[0]
            x_new   = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

            cal_next = (future_exog[i] if future_exog.ndim > 1
                        else np.array([future_exog[i]], dtype=np.float32))
            cal_lb   = cal_window if include_cal_lookback else None

            x_new[0] = torch.tensor(
                generate_node_features(
                    target_window_scaled, cal_next=cal_next, cal_lookback=cal_lb,
                    selected_features=node_features, cal_columns=cal_columns,
                ),
                dtype=torch.float32,
            )
            for node_idx in range(1, n_nodes):
                orig_feat   = G_data.x[node_idx].numpy()
                neighbor_ts = orig_feat[:graph_window_size]
                x_new[node_idx] = torch.tensor(
                    generate_node_features(
                        neighbor_ts, selected_features=node_features,
                        is_neighbor=True, pad_ts_to=lookback, cal_columns=cal_columns,
                    ),
                    dtype=torch.float32,
                )
            G_data.x = x_new

            # ── 3. Build ts_seq for LSTM: ts_seq[t] = (value_t, cal_{t+1}) ─
            if cal_dim > 0:
                _cal_shifted = np.vstack([
                    rolling_cal[1:],
                    cal_next.reshape(1, -1),
                ])  # (lookback, cal_dim)
                ts_seq_np = np.concatenate(
                    [target_window_scaled[:, None], _cal_shifted], axis=1
                )  # (lookback, 1+cal_dim)
            else:
                ts_seq_np = target_window_scaled[:, None]
            ts_seq = torch.from_numpy(ts_seq_np).float().unsqueeze(0).to(device)

            # ── 4. Forward pass ─────────────────────────────────────────────
            pyg_batch           = Batch.from_data_list([G_data]).to(device)
            target_node_indices = torch.tensor([0], dtype=torch.long, device=device)
            y_pred   = model(pyg_batch, target_node_indices, ts_seq)
            val_pred = y_pred.reshape(-1)[0].item()

            unscaled_val = scaler.inverse_transform([[val_pred]])[0, 0]
            preds_unscaled.append(unscaled_val)

            # ── 5. Shift rolling windows ─────────────────────────────────────
            target_window_scaled = np.roll(target_window_scaled, -1)
            target_window_scaled[-1] = val_pred
            if cal_dim > 0:
                rolling_cal = np.roll(rolling_cal, -1, axis=0)
                rolling_cal[-1] = cal_next
            if include_cal_lookback:
                cal_window = np.roll(cal_window, -1, axis=0)
                cal_window[-1] = cal_next

    return np.array(preds_unscaled).flatten()


# ---------------------------------------------------------------------------
# GNN-at-every-step LSTM recursive inference  (SimpleGNNSeqLSTMForecaster)
# ---------------------------------------------------------------------------

def recursive_inference_seq(
    model: torch.nn.Module,
    scaler,
    recent_history: np.ndarray,
    future_exog: np.ndarray,
    target_channel: int = 0,
    device: Optional[str] = None,
    # --- Graph parameters ---
    df_wide: pd.DataFrame = None,
    cat_labels: dict = None,
    target_id: int = None,
    metric: str = 'spearman',
    fixed_threshold: float = 0.5,
    enable_edges_within_star: bool = False,
    enable_second_degree: bool = False,
    past_dates: list = None,
    future_dates: list = None,
    graph_window_size: int = 15,
    node_features=None,
) -> np.ndarray:
    """
    Recursive 1-step-ahead inference for SimpleGNNSeqLSTMForecaster.

    At each forecast step the model receives:
      * L ego-graphs (one per lookback step, 15-day windows)
      * ts_seq (B=1, L, 1+cal_dim) — (value_t, cal_{t+1})

    Strategy: maintain a rolling deque of L graph base structures.
      • Initial deque: L historical graphs built from past data.
      • After each prediction: pop the oldest graph, build 1 new graph
        for the next position (using predictions for the target where
        the window overlaps with future dates), then push it.
      • At every step only the target-node features are recomputed from
        the current `target_window_scaled`; neighbor features stay fixed.

    Parameters match recursive_inference_pure_sage (same signature).
    """
    from collections import deque
    from copy import copy
    from torch_geometric.data import Batch
    from gnn_lstm_pyg import (compute_target_node_features_seq,
                               compute_neighbor_node_features_pure)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    C_in         = recent_history.shape[1]
    lookback     = recent_history.shape[0]
    exog_indices = [i for i in range(C_in) if i != target_channel]
    cal_dim      = len(exog_indices)
    horizon      = len(future_exog)

    x_scaled = recent_history.copy()
    x_scaled[:, target_channel:target_channel+1] = scaler.transform(
        recent_history[:, target_channel:target_channel+1]
    )

    target_window_scaled = x_scaled[:, target_channel].copy()    # (lookback,)
    rolling_cal = x_scaled[:, exog_indices].copy() if cal_dim > 0 else None

    all_dates   = list(past_dates) + list(future_dates)
    past_T      = len(past_dates)
    feature_dim = graph_window_size + cal_dim + 8

    # ── Helper: build one graph and override target-node features ────────────
    def _build_graph(t_global, ts_window_scaled, cal_at_t):
        """Build ego-graph for window ending at all_dates[t_global]."""
        date_start_idx = max(0, t_global - graph_window_size + 1)
        window_dates   = all_dates[date_start_idx : t_global + 1]

        # For future dates in the window, substitute predictions into df_wide copy
        target_preds_for_graph = df_wide.loc[target_id, window_dates].values.astype(float).copy()
        for k, d in enumerate(window_dates):
            if d in future_dates:
                fut_idx = list(future_dates).index(d)
                if fut_idx < len(preds_unscaled):
                    target_preds_for_graph[k] = preds_unscaled[fut_idx]

        G = build_dynamic_graph_with_calculated_threshold(
            target_id=target_id, target_preds=target_preds_for_graph,
            df_wide=df_wide, cat_labels=cat_labels, date_cols=window_dates,
            metric=metric, fixed_threshold=fixed_threshold,
            enable_edges_within_star=enable_edges_within_star,
            enable_second_degree=enable_second_degree,
            node_features=node_features,
        )

        n_nodes = G.x.shape[0]
        x_new   = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

        # Target node: use the provided scaled ts window
        x_new[0] = torch.tensor(
            compute_target_node_features_seq(ts_window_scaled, cal_at_t, feature_dim),
            dtype=torch.float32,
        )
        # Neighbor nodes: reuse raw ts from the graph (fixed)
        for node_idx in range(1, n_nodes):
            orig_feat   = G.x[node_idx].numpy()
            neighbor_ts = orig_feat[:graph_window_size]
            x_new[node_idx] = torch.tensor(
                compute_neighbor_node_features_pure(neighbor_ts, feature_dim),
                dtype=torch.float32,
            )
        G.x = x_new
        G.central_node_idx = 0
        return G

    # ── Build initial buffer of L historical graphs ───────────────────────────
    preds_unscaled = []   # filled incrementally so _build_graph can reference it
    graph_buffer   = deque()

    # We also maintain a (lookback + graph_window_size - 1) ts history so we can
    # extract the right window for each graph in the buffer.
    # ts_history[-1] = most recent; ts_history[-lookback:] = current window.
    history_len = lookback + graph_window_size - 1
    ts_history  = np.zeros(history_len, dtype=np.float32)
    # Fill from scaled history (pad left with first value if needed)
    fill_start = max(0, history_len - lookback)
    ts_history[fill_start:] = target_window_scaled[:history_len - fill_start]

    for l in range(lookback):
        t_global = past_T - lookback + l          # position in all_dates
        # ts window of graph_window_size ending at t_global (from ts_history)
        hist_idx  = fill_start + l                # position in ts_history
        win_start = max(0, hist_idx - graph_window_size + 1)
        ts_win    = ts_history[win_start : hist_idx + 1]
        cal_at_t  = x_scaled[l, exog_indices] if cal_dim > 0 else np.array([], dtype=np.float32)
        graph_buffer.append(_build_graph(t_global, ts_win, cal_at_t))

    model = model.to(device).eval()

    with torch.no_grad():
        for i in range(horizon):
            cal_next = (future_exog[i] if future_exog.ndim > 1
                        else np.array([future_exog[i]], dtype=np.float32))

            # ── Override target-node features in ALL buffer graphs ───────────
            # Each buffer graph l uses a ts window ending at (past_T - L + i + l).
            # We recompute only the target-node features; neighbor features stay.
            updated_graphs = []
            for l, G_base in enumerate(graph_buffer):
                # Rolling ts position: lookback step l corresponds to
                # target_window_scaled[l] in the current window.
                hist_win_end   = l   # index into target_window_scaled
                win_start      = max(0, hist_win_end - graph_window_size + 1)
                ts_win_l       = target_window_scaled[win_start : hist_win_end + 1]
                cal_at_l       = rolling_cal[l] if cal_dim > 0 else np.array([], dtype=np.float32)

                n_nodes = G_base.x.shape[0]
                x_new   = G_base.x.clone()
                x_new[0] = torch.tensor(
                    compute_target_node_features_seq(ts_win_l, cal_at_l, feature_dim),
                    dtype=torch.float32,
                )
                G_upd = copy(G_base)
                G_upd.x = x_new
                updated_graphs.append(G_upd)

            # ── Build ts_seq: [value_t | cal_{t+1}] for t=0..L-1 ────────────
            if cal_dim > 0:
                cal_shifted = np.vstack([rolling_cal[1:], cal_next.reshape(1, -1)])
                ts_seq_np   = np.concatenate(
                    [target_window_scaled[:, None], cal_shifted], axis=1
                )
            else:
                ts_seq_np = target_window_scaled[:, None]
            ts_seq = torch.from_numpy(ts_seq_np).float().unsqueeze(0).to(device)

            # ── Forward pass ─────────────────────────────────────────────────
            flat_graphs = updated_graphs                         # list of L graphs
            pyg_batch   = Batch.from_data_list(flat_graphs).to(device)
            y_pred   = model(ts_seq, pyg_batch, B=1, L=lookback)
            val_pred = y_pred.reshape(-1)[0].item()

            unscaled_val = scaler.inverse_transform([[val_pred]])[0, 0]
            preds_unscaled.append(unscaled_val)

            # ── Shift rolling windows ─────────────────────────────────────────
            target_window_scaled = np.roll(target_window_scaled, -1)
            target_window_scaled[-1] = val_pred
            if cal_dim > 0:
                rolling_cal = np.roll(rolling_cal, -1, axis=0)
                rolling_cal[-1] = cal_next

            # ── Build new graph for the position sliding in ───────────────────
            if i < horizon - 1:
                t_new      = past_T + i               # next future position
                ts_win_new = target_window_scaled[-graph_window_size:]
                cal_new    = rolling_cal[-1] if cal_dim > 0 else np.array([], dtype=np.float32)
                new_graph  = _build_graph(t_new, ts_win_new, cal_new)
                graph_buffer.popleft()
                graph_buffer.append(new_graph)

    return np.array(preds_unscaled).flatten()

