import torch
import numpy as np
from typing import Optional
from utils import compute_distances_1vsAll, compute_similarities_1vsAll
import networkx as nx
import pandas as pd
from torch_geometric.data import Data
def build_dynamic_graph_with_calculated_threshold(target_id, target_preds, df_wide, cat_labels, date_cols, metric, fixed_threshold, enable_edges_within_star=True, enable_second_degree=False, node_features: list = None,
                                                  node_scalers: dict = None):
    
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

    # Build feature matrix X (using window timeseries as features)
    x_matrix = []
    node_ts_matrix = []
    _feats = node_features if node_features is not None else ['ts', 'last_demand', 'mean7', 'mean_all', 'std_all', 'zero_ratio', 'slope', 'min_v', 'max_v']
    _storage_feats = list(_feats)
    for n_id in graph_nodes:
        # Special logic for the central node because its current values are predictions
        if n_id == target_id:
            row = np.asarray(target_ts, dtype=np.float32)
        else:
            row = window_data.loc[n_id].values.astype(np.float32)

        # Apply per-node train-fit scaler when available (consistent with target).
        # Skip scaling for the target node here because the caller (recursive
        # inference) overrides target node features afterwards with its own
        # already-scaled rolling window.
        if n_id != target_id and node_scalers is not None and n_id in node_scalers:
            row_scaled = node_scalers[n_id].transform(row.reshape(-1, 1)).flatten().astype(np.float32)
        elif n_id != target_id:
            r_min, r_max = float(row.min()), float(row.max())
            r_range = r_max - r_min
            row_scaled = ((row - r_min) / r_range).astype(np.float32) if r_range > 1e-8 else np.zeros_like(row)
        else:
            row_scaled = row

        node_ts_matrix.append(row_scaled)
        features = generate_node_features(row_scaled, selected_features=_storage_feats)
        x_matrix.append(features)
        
    x = torch.tensor(np.array(x_matrix), dtype=torch.float)
    node_ts_tensor = torch.tensor(np.array(node_ts_matrix), dtype=torch.float)

    # PyG Data Object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_ts  = node_ts_tensor
    data.node_ids = list(graph_nodes)
    data.central_node_idx = 0
    data.target_id = target_id
    
    return data

def recursive_inference_gcn_mlp(
    model: torch.nn.Module,
    scaler,
    recent_history: np.ndarray,
    future_exog: np.ndarray,
    target_channel: int = 0,
    device: Optional[str] = None,
    # --- GCN Parameters ---
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
    node_features: list = None,
    cal_columns: list = None,
    node_scalers: dict = None,
    # --- Leak-safe dynamic exog (optional) ---
    exog_scaler=None,
    exog_cols: list = None,
    history_exog_unscaled: np.ndarray = None,   # (lookback, n_exog) UNSCALED
    future_exog_unscaled: np.ndarray = None,    # (horizon, n_exog) UNSCALED
) -> np.ndarray:
    """
    Recursive 1-step-ahead inference for the GCNMLPForecaster.

    At each autoregressive step:
      1. Build ONE ego-graph using the last `graph_window_size` dates.
      2. Override the target node's features with the full rolling lookback
         window (scaled) + next-step calendar features.
      3. Forward pass → 1-step prediction → unscale → append → shift window.

    If `exog_scaler`, `exog_cols` and `future_exog_unscaled` are provided, the
    function operates in LEAK-SAFE mode: every `lag_*` and
    `rolling_mean_excl_*` column is recomputed each step from the running
    buffer of (past + predicted) UNSCALED target values and re-scaled with
    `exog_scaler`. In that mode the `future_exog` argument is ignored.
    """
    from torch_geometric.data import Batch
    from gnn_pyg import generate_node_features
    from utils import parse_dynamic_exog_cols

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dynamic_mode = (exog_scaler is not None and exog_cols is not None
                    and future_exog_unscaled is not None)
    if dynamic_mode:
        exog_cols = list(exog_cols)
        lag_cols_d, roll_cols_d = parse_dynamic_exog_cols(exog_cols)
        col_idx = {c: i for i, c in enumerate(exog_cols)}
        future_exog_unscaled = np.asarray(future_exog_unscaled, dtype=np.float64).copy()

    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    C_in            = recent_history.shape[1]
    lookback        = recent_history.shape[0]
    exog_indices    = [i for i in range(C_in) if i != target_channel]
    cal_dim         = len(exog_indices)
    horizon         = (len(future_exog_unscaled) if dynamic_mode else len(future_exog))

    # Running buffer of UNSCALED past targets (history + preds). Used in
    # dynamic mode to recompute lag/rolling features each step.
    target_buffer_unscaled = list(
        recent_history[:, target_channel].astype(np.float64).ravel()
    )

    # Scale the target channel; keep exog as-is (already scaled by caller)
    x_scaled = recent_history.copy()
    x_scaled[:, target_channel:target_channel+1] = scaler.transform(
        recent_history[:, target_channel:target_channel+1]
    )

    # Rolling target window (scaled) used for node features
    target_window_scaled = x_scaled[:, target_channel].copy()   # (lookback,)

    all_dates        = list(past_dates) + list(future_dates)

    if include_cal_lookback:
        # Rolling calendar lookback window (unscaled exog already scaled by caller)
        cal_window   = x_scaled[:, exog_indices].copy()  # (lookback, cal_dim)
    else:
        cal_window   = None

    # Compute feature_dim dynamically from the actual feature list.
    _dummy_ts  = np.zeros(lookback, dtype=np.float32)
    _dummy_cal = np.zeros(cal_dim, dtype=np.float32) if cal_dim > 0 else None
    _dummy_lb  = (np.zeros((lookback, cal_dim), dtype=np.float32)
                  if include_cal_lookback and cal_dim > 0 else None)
    feature_dim = len(generate_node_features(
        _dummy_ts, cal_next=_dummy_cal, cal_lookback=_dummy_lb,
        selected_features=node_features, cal_columns=cal_columns,
    ))

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
                node_features=node_features,
                node_scalers=node_scalers,
            )

            # ---- 2. Override node features with enhanced representations ----
            n_nodes = G_data.x.shape[0]
            x_new   = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

            # Build next-step calendar features:
            #   - dynamic mode: recompute lag/rolling from target_buffer_unscaled
            #     and re-scale with exog_scaler.
            #   - static mode: use the precomputed future_exog row as-is (assumed
            #     already scaled by caller).
            if dynamic_mode:
                row_unscaled = future_exog_unscaled[i].copy()
                for col, k in lag_cols_d.items():
                    idx = col_idx[col]
                    row_unscaled[idx] = (target_buffer_unscaled[-k]
                                          if k <= len(target_buffer_unscaled) else 0.0)
                for col, w in roll_cols_d.items():
                    idx = col_idx[col]
                    vals = (target_buffer_unscaled[-w:]
                            if w <= len(target_buffer_unscaled) else target_buffer_unscaled)
                    row_unscaled[idx] = float(np.mean(vals)) if len(vals) > 0 else 0.0
                cal_next = exog_scaler.transform(
                    row_unscaled.reshape(1, -1)
                ).ravel().astype(np.float32)
            else:
                cal_next = (future_exog[i] if future_exog.ndim > 1
                            else np.array([future_exog[i]], dtype=np.float32))

            cal_lb   = cal_window if include_cal_lookback else None
            
            selected_target = node_features 
            x_new[0] = torch.tensor(
                generate_node_features(target_window_scaled, cal_next=cal_next, cal_lookback=cal_lb,
                                       selected_features=selected_target, cal_columns=cal_columns),
                dtype=torch.float32,
            )

            # Neighbor nodes: pad to feature_dim.
            # Read neighbor ts from the explicit node_ts tensor (already scaled
            # with the per-product train-fit scaler when node_scalers was given).
            _dummy_cal_next = np.zeros(cal_dim, dtype=np.float32) if cal_dim > 0 else None
            for node_idx in range(1, n_nodes):
                if hasattr(G_data, 'node_ts') and G_data.node_ts is not None:
                    neighbor_ts = G_data.node_ts[node_idx].numpy()
                else:
                    # Backward-compatible fallback: legacy convention where the
                    # first graph_window_size entries of x are the raw ts.
                    orig_feat   = G_data.x[node_idx].numpy()
                    neighbor_ts = orig_feat[:graph_window_size]
                    n_min, n_max = neighbor_ts.min(), neighbor_ts.max()
                    n_range = n_max - n_min
                    if n_range > 1e-8:
                        neighbor_ts = (neighbor_ts - n_min) / n_range
                    else:
                        neighbor_ts = np.zeros_like(neighbor_ts)

                selected_neighbor = node_features
                x_new[node_idx] = torch.tensor(
                    generate_node_features(neighbor_ts, cal_next=_dummy_cal_next,
                                           selected_features=selected_neighbor,
                                           is_neighbor=True, pad_ts_to=lookback, cal_columns=cal_columns),
                    dtype=torch.float32,
                )

            G_data.x = x_new

            # ---- 3. Forward pass ----
            pyg_batch           = Batch.from_data_list([G_data]).to(device)
            target_node_indices = torch.tensor([0], dtype=torch.long, device=device)

            # Build ts_seq for the MLP branch: (1, lookback, 1 + cal_dim)
            _ts_t = torch.tensor(target_window_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
            if cal_dim > 0 and cal_window is not None:
                _ts_c = torch.tensor(cal_window, dtype=torch.float32).unsqueeze(0)  # (1, L, cal_dim)
                ts_seq = torch.cat([_ts_t, _ts_c], dim=-1).to(device)              # (1, L, 1+cal_dim)
            else:
                ts_seq = _ts_t.to(device)                                           # (1, L, 1)

            y_pred    = model(pyg_batch, target_node_indices, ts_seq)
            val_pred  = y_pred.view(-1)[0].item()

            unscaled_val = scaler.inverse_transform([[val_pred]])[0, 0]
            preds_unscaled.append(unscaled_val)
            target_buffer_unscaled.append(float(unscaled_val))

            # ---- 4. Shift rolling windows ----
            target_window_scaled = np.roll(target_window_scaled, -1)
            target_window_scaled[-1] = val_pred

            # Shift calendar window: the step we just predicted contributes cal_next
            if include_cal_lookback:
                cal_window = np.roll(cal_window, -1, axis=0)
                cal_window[-1] = cal_next

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
    Recursive inference for GCNMLPForecaster with no graph (dummy single-node graph).
    Mirrors the no-graph training path: a single isolated node with no edges is used
    so the GCNConv passes through only the self-loop, and the MLP branch drives predictions.
    """
    from torch_geometric.data import Batch, Data

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    horizon = len(future_exog)
    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    C_in         = recent_history.shape[1]
    lookback     = recent_history.shape[0]
    exog_indices = [idx for idx in range(C_in) if idx != target_channel]
    cal_dim      = len(exog_indices)

    x_scaled = recent_history.copy()
    x_scaled[:, target_channel:target_channel+1] = scaler.transform(
        recent_history[:, target_channel:target_channel+1]
    )

    target_window_scaled = x_scaled[:, target_channel].copy()              # (lookback,)
    cal_window           = x_scaled[:, exog_indices].copy() if cal_dim > 0 else None  # (lookback, cal_dim)

    # Infer node feature dimension from the model's first GCNConv layer
    node_feat_dim = model.conv1.in_channels

    preds_unscaled = []
    model = model.to(device).eval()

    with torch.no_grad():
        for i in range(horizon):
            # ---- Build ts_seq: (1, lookback, 1 + cal_dim) ----
            _ts_t = torch.tensor(target_window_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            if cal_dim > 0 and cal_window is not None:
                _ts_c = torch.tensor(cal_window, dtype=torch.float32).unsqueeze(0)
                ts_seq = torch.cat([_ts_t, _ts_c], dim=-1).to(device)
            else:
                ts_seq = _ts_t.to(device)

            # ---- Dummy single-node graph (no edges) ----
            dummy_x    = torch.zeros((1, node_feat_dim), dtype=torch.float32)
            dummy_graph = Data(
                x=dummy_x,
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1), dtype=torch.float),
            )
            pyg_batch           = Batch.from_data_list([dummy_graph]).to(device)
            target_node_indices = torch.tensor([0], dtype=torch.long, device=device)

            y_pred   = model(pyg_batch, target_node_indices, ts_seq)
            val_pred = y_pred.view(-1)[0].item()

            unscaled_val = scaler.inverse_transform([[val_pred]])[0, 0]
            preds_unscaled.append(unscaled_val)

            # ---- Shift rolling windows ----
            target_window_scaled = np.roll(target_window_scaled, -1)
            target_window_scaled[-1] = val_pred
            if cal_dim > 0 and cal_window is not None:
                cal_next   = (future_exog[i] if future_exog.ndim > 1
                              else np.array([future_exog[i]], dtype=np.float32))
                cal_window = np.roll(cal_window, -1, axis=0)
                cal_window[-1] = cal_next

    return np.array(preds_unscaled).flatten()
