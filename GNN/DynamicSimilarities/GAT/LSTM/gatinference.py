import torch
import numpy as np
from typing import Optional
from utils import compute_distances_1vsAll, compute_similarities_1vsAll
import networkx as nx
import pandas as pd
def build_dynamic_graph_with_calculated_threshold(target_id, target_preds, df_wide, cat_labels, date_cols, metric, fixed_threshold, enable_edges_within_star=True, enable_second_degree=False):
    from torch_geometric.data import Data
    try:
        from graphsage_pyg import compute_node_features
    except ImportError:
        # Fallback if compute_node_features is not available in the inference directory
        compute_node_features = lambda x: x

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

    # Build feature matrix X (using window timeseries as features)
    x_matrix = []
    for n_id in graph_nodes:
        # Special logic for the central node because its current values are predictions
        if n_id == target_id:
            features = compute_node_features(target_ts)
        else:
            row = window_data.loc[n_id].values
            features = compute_node_features(row)
        x_matrix.append(features)
        
    x = torch.tensor(np.array(x_matrix), dtype=torch.float)

    # PyG Data Object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.central_node_idx = 0
    data.target_id = target_id
    
    return data

def recursive_inference_pure_sage(
    model: torch.nn.Module,
    scaler,
    recent_history: np.ndarray,
    future_exog: np.ndarray,
    target_channel: int = 0,
    device: Optional[str] = None,
    # --- GraphSAGE Parameters ---
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
) -> np.ndarray:
    """
    Recursive 1-step-ahead inference for the PureGraphSAGEForecaster.

    At each autoregressive step:
      1. Build ONE ego-graph using the last `graph_window_size` dates.
      2. Override the target node's features with the full rolling lookback
         window (scaled) + next-step calendar features.
      3. Forward pass → 1-step prediction → unscale → append → shift window.
    """
    from torch_geometric.data import Batch
    from graphsage_pyg import compute_target_node_features_pure, compute_neighbor_node_features_pure

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    C_in            = recent_history.shape[1]
    lookback        = recent_history.shape[0]
    exog_indices    = [i for i in range(C_in) if i != target_channel]
    cal_dim         = len(exog_indices)
    horizon         = len(future_exog)

    # Scale the target channel; keep exog as-is (already scaled by caller)
    x_scaled = recent_history.copy()
    x_scaled[:, target_channel:target_channel+1] = scaler.transform(
        recent_history[:, target_channel:target_channel+1]
    )

    # Rolling target window (scaled) used for node features
    target_window_scaled = x_scaled[:, target_channel].copy()   # (lookback,)

    all_dates        = list(past_dates) + list(future_dates)
    feature_dim      = lookback + cal_dim + 8

    preds_unscaled   = []
    model            = model.to(device).eval()

    with torch.no_grad():
        for i in range(horizon):
            # ---- 1. Build one ego-graph for the current window ----
            # The graph window ends at the last *observed* timestep
            t_idx_global = len(past_dates) - 1 + i
            date_start   = max(0, t_idx_global - graph_window_size + 1)
            window_dates = all_dates[date_start : t_idx_global + 1]

            # Baseline target values from df_wide; overwrite with predictions
            target_preds_for_graph = (
                df_wide.loc[target_id, window_dates].values.astype(float).copy()
            )
            for f_idx, f_date in enumerate(future_dates[:i]):
                if f_date in window_dates:
                    w_idx = list(window_dates).index(f_date)
                    target_preds_for_graph[w_idx] = preds_unscaled[f_idx]

            G_data = build_dynamic_graph_with_calculated_threshold(
                target_id=target_id,
                target_preds=target_preds_for_graph,
                df_wide=df_wide,
                cat_labels=cat_labels,
                date_cols=window_dates,
                metric=metric,
                fixed_threshold=fixed_threshold,
                enable_edges_within_star=enable_edges_within_star,
                enable_second_degree=enable_second_degree,
            )

            # ---- 2. Override node features with enhanced representations ----
            n_nodes = G_data.x.shape[0]
            x_new   = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

            # Target node: full scaled lookback + next-step calendar + stats
            cal_next = (future_exog[i] if future_exog.ndim > 1
                        else np.array([future_exog[i]], dtype=np.float32))
            x_new[0] = torch.tensor(
                compute_target_node_features_pure(target_window_scaled, cal_next, feature_dim),
                dtype=torch.float32,
            )

            # Neighbor nodes: pad to feature_dim
            for node_idx in range(1, n_nodes):
                orig_feat   = G_data.x[node_idx].numpy()
                neighbor_ts = orig_feat[:graph_window_size]
                x_new[node_idx] = torch.tensor(
                    compute_neighbor_node_features_pure(neighbor_ts, feature_dim),
                    dtype=torch.float32,
                )

            G_data.x = x_new

            # ---- 3. Forward pass ----
            pyg_batch           = Batch.from_data_list([G_data]).to(device)
            target_node_indices = torch.tensor([0], dtype=torch.long, device=device)

            y_pred    = model(pyg_batch, target_node_indices)
            val_pred  = y_pred.item()

            unscaled_val = scaler.inverse_transform([[val_pred]])[0, 0]
            preds_unscaled.append(unscaled_val)

            # ---- 4. Shift rolling target window ----
            target_window_scaled = np.roll(target_window_scaled, -1)
            target_window_scaled[-1] = val_pred

    return np.array(preds_unscaled).flatten()


def recursive_inference_sage_lstm(
    model: torch.nn.Module,
    scaler,
    recent_history: np.ndarray,
    future_exog: np.ndarray,
    target_channel: int = 0,
    device: Optional[str] = None,
    # --- GraphSAGE Parameters ---
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
) -> np.ndarray:
    """
    Recursive 1-step-ahead inference for GraphSAGELSTMForecaster.

    At each autoregressive step i, L ego-graphs are built (one per lookback
    position).  The target-node features of each graph are overridden with the
    per-step windowed ts + next-step calendar (compute_target_node_features_seq).
    All L graphs are forwarded in a single batch → LSTM → 1-step prediction.

    Calendar convention: extended_cal[l + i + 1] is the next-step-shifted
    calendar for lookback position l at inference step i, matching the
    alignment used in make_sequence_windows.
    """
    from torch_geometric.data import Batch
    from graphsage_pyg import (compute_target_node_features_seq,
                                compute_neighbor_node_features_pure)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    C_in         = recent_history.shape[1]
    lookback     = recent_history.shape[0]
    exog_indices = [idx for idx in range(C_in) if idx != target_channel]
    cal_dim      = len(exog_indices)
    horizon_len  = len(future_exog)

    # Scale the target channel; exog is already scaled by the caller
    x_scaled = recent_history.copy()
    x_scaled[:, target_channel:target_channel+1] = scaler.transform(
        recent_history[:, target_channel:target_channel+1]
    )

    target_window_scaled = x_scaled[:, target_channel].copy()   # (lookback,)

    # Extended calendar: past (lookback rows) + future (horizon rows)
    # extended_cal[l + i + 1] = next-step cal for lookback pos l at step i
    past_cal = x_scaled[:, exog_indices] if cal_dim > 0 else np.zeros((lookback, 0), dtype=np.float32)
    extended_cal = (np.vstack([past_cal, future_exog])
                    if cal_dim > 0 else np.zeros((lookback + horizon_len, 0), dtype=np.float32))

    all_dates   = list(past_dates) + list(future_dates)
    feature_dim = graph_window_size + cal_dim + 8

    preds_unscaled = []
    model = model.to(device).eval()

    with torch.no_grad():
        for i in range(horizon_len):
            seq_graphs = []

            for l in range(lookback):
                # Global date index for lookback position l at step i
                t_global   = len(past_dates) - lookback + i + l
                date_start = max(0, t_global - graph_window_size + 1)
                window_dates = all_dates[date_start : t_global + 1]

                # Baseline target ts from df_wide; overwrite future dates with predictions
                target_preds_for_graph = (
                    df_wide.loc[target_id, window_dates].values.astype(float).copy()
                )
                for f_idx, f_date in enumerate(future_dates[:i]):
                    if f_date in window_dates:
                        w_idx = list(window_dates).index(f_date)
                        target_preds_for_graph[w_idx] = preds_unscaled[f_idx]

                G_data = build_dynamic_graph_with_calculated_threshold(
                    target_id=target_id,
                    target_preds=target_preds_for_graph,
                    df_wide=df_wide,
                    cat_labels=cat_labels,
                    date_cols=window_dates,
                    metric=metric,
                    fixed_threshold=fixed_threshold,
                    enable_edges_within_star=enable_edges_within_star,
                    enable_second_degree=enable_second_degree,
                )

                # ── Override node features ──
                n_nodes = G_data.x.shape[0]
                x_new   = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

                # Target node: scaled ts slice from rolling window + next-step cal
                ts_win   = target_window_scaled[max(0, l - graph_window_size + 1) : l + 1]
                cal_next = extended_cal[l + i + 1]   # always in-bounds (max = lookback+horizon-1)

                x_new[0] = torch.tensor(
                    compute_target_node_features_seq(ts_win, cal_next, feature_dim),
                    dtype=torch.float32,
                )

                # Neighbor nodes
                for node_idx in range(1, n_nodes):
                    orig_feat   = G_data.x[node_idx].numpy()
                    neighbor_ts = orig_feat[:graph_window_size]
                    x_new[node_idx] = torch.tensor(
                        compute_neighbor_node_features_pure(neighbor_ts, feature_dim),
                        dtype=torch.float32,
                    )

                G_data.x = x_new
                seq_graphs.append(G_data)

            # ── Forward pass with all L graphs ──
            pyg_batch           = Batch.from_data_list(seq_graphs).to(device)
            target_node_indices = pyg_batch.ptr[:-1]   # (L,) — node 0 per graph

            y_pred   = model(pyg_batch, target_node_indices, B=1, L=lookback)
            val_pred = y_pred.item()

            unscaled_val = scaler.inverse_transform([[val_pred]])[0, 0]
            preds_unscaled.append(unscaled_val)

            # ── Shift rolling target window ──
            target_window_scaled = np.roll(target_window_scaled, -1)
            target_window_scaled[-1] = val_pred

    return np.array(preds_unscaled).flatten()


def recursive_inference_no_graph(
    model: torch.nn.Module,
    scaler,
    recent_history: np.ndarray,
    future_exog: np.ndarray,
    target_channel: int = 0,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Pure-MLP recursive inference (no GraphSAGE, no graph inputs).
    Inputs: only target value + exogenous calendar features.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    horizon = len(future_exog)
    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    C_in = recent_history.shape[1]
    exog_indices = [idx for idx in range(C_in) if idx != target_channel]

    current_x_scaled = recent_history.copy()
    current_x_scaled[:, target_channel:target_channel+1] = scaler.transform(
        recent_history[:, target_channel:target_channel+1]
    )
    input_window = current_x_scaled.copy()

    # Align exogenous features forward by 1 (same convention as training)
    if len(exog_indices) > 0:
        input_window[:-1, exog_indices] = input_window[1:, exog_indices]
        input_window[-1, exog_indices] = future_exog[0]

    preds_unscaled = []
    model = model.to(device).eval()

    with torch.no_grad():
        for i in range(horizon):
            x_tensor = torch.from_numpy(input_window).float().unsqueeze(0).to(device)  # (1, L, C)
            y_pred = model(x_tensor)
            val_pred = y_pred.view(-1)[0].item()

            unscaled_val = scaler.inverse_transform([[val_pred]])[0, 0]
            preds_unscaled.append(unscaled_val)

            input_window = np.roll(input_window, -1, axis=0)
            input_window[-1, target_channel] = val_pred
            if len(exog_indices) > 0 and (i + 1) < horizon:
                input_window[-1, exog_indices] = future_exog[i + 1]

    return np.array(preds_unscaled).flatten()
