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

def recursive_inference(
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
    metric: str = 'cid',
    fixed_threshold: float = 0.5,
    enable_edges_within_star: bool = False,
    enable_second_degree: bool = False,
    past_dates: list = None,
    future_dates: list = None,
    graph_window_size: int = 15,
) -> np.ndarray:
    """
    Recursively forecasts `horizon` steps using a 1-step-ahead GraphSAGE+MLP model.
    Uses known `future_exog` for the next step's input.
    """
    from torch_geometric.data import Batch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    horizon = len(future_exog)
    
    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    C_in = recent_history.shape[1]
    exog_indices = [idx for idx in range(C_in) if idx != target_channel]
    ts_indices = [target_channel]

    current_x_scaled = recent_history.copy()
    current_x_scaled[:, target_channel:target_channel+1] = scaler.transform(
        recent_history[:, target_channel:target_channel+1]
    )
    
    lookback = current_x_scaled.shape[0]
    
    # We maintain the window of scaled features
    input_window = current_x_scaled.copy()
    
    # We also need to maintain the true, unscaled targets to build graphs
    all_unscaled_targets = list(recent_history[:, target_channel])
    all_dates = list(past_dates) + list(future_dates)
    
    preds_scaled = []
    preds_unscaled = []
    model = model.to(device).eval()

    # Align exogenous features forward for t+1 prediction inside the sequence
    if len(exog_indices) > 0:
        input_window[:-1, exog_indices] = input_window[1:, exog_indices]
        input_window[-1, exog_indices] = future_exog[0]

    with torch.no_grad():
        for i in range(horizon):
            
            # --- 1. Graph Building for the lookback sequence ---
            # We must re-build the sequence of L PyG graphs since target predictions shift
            graphs_seq = []
            
            for step in range(lookback):
                # Global date index (into past_dates + future_dates)
                t_idx_global = len(past_dates) - lookback + step + i

                # Clamp window starts so we never produce empty slices
                date_start = max(0, t_idx_global - graph_window_size + 1)
                window_dates = all_dates[date_start : t_idx_global + 1]

                # Fetch baseline unscaled values directly from df_wide
                target_preds_for_graph = df_wide.loc[target_id, window_dates].values.astype(float).copy()
                
                # Overwrite any future dates in the window with our autoregressive predictions
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
                    enable_second_degree=enable_second_degree
                )
                graphs_seq.append(G_data)

            # --- 2. Inference Prep ---
            ts_seq = input_window[:, ts_indices]
            cal_seq = input_window[:, exog_indices]
            
            ts_tensor = torch.from_numpy(ts_seq).float().unsqueeze(0).to(device)  # (1, L, ts_dim)
            cal_tensor = torch.from_numpy(cal_seq).float().unsqueeze(0).to(device) # (1, L, cal_dim)
            
            # Create a PyG Batch for this specific single-sequence window
            pyg_batch = Batch.from_data_list(graphs_seq).to(device)
            target_node_indices = pyg_batch.ptr[:-1] # Because target_id was forced to node 0 in each Data object
            
            # --- 3. Forward Pass ---
            y_pred = model(ts_tensor, cal_tensor, pyg_batch, target_node_indices)
            val_pred = y_pred.item()
            preds_scaled.append(val_pred)
            
            unscaled_val = scaler.inverse_transform([[val_pred]])[0, 0]
            preds_unscaled.append(unscaled_val)
            all_unscaled_targets.append(unscaled_val)

            # --- 4. Shift logic (for next step t+1) ---
            input_window = np.roll(input_window, -1, axis=0)
            
            input_window[-1, target_channel] = val_pred
            
            if len(exog_indices) > 0 and (i + 1) < horizon:
                input_window[-1, exog_indices] = future_exog[i + 1]
            
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
