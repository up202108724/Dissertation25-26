import torch
import numpy as np
from typing import Optional
from utils import compute_distances_1vsAll, compute_similarities_1vsAll
import networkx as nx
import pandas as pd
import time
import os
def build_dynamic_graph_with_calculated_threshold(target_id, target_preds, df_wide, cat_labels, date_cols, metric, fixed_threshold, enable_edges_within_star=True, enable_second_degree=False, node_features: list = None):
    from torch_geometric.data import Data
    try:
        from graphsage_pyg import generate_node_features
    except ImportError:
        # Fallback if generate_node_features is not available in the inference directory
        generate_node_features = lambda x, **kwargs: x

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
    _feats = node_features if node_features is not None else ['ts', 'last_demand', 'mean7', 'mean_all', 'std_all', 'zero_ratio', 'slope', 'min_v', 'max_v']
    # Always store raw ts FIRST so make_single_windows can reliably extract
    # the neighbor ts via orig_feat[:graph_window_size].
    _storage_feats = ['ts'] + [f for f in _feats if f != 'ts']
    for n_id in graph_nodes:
        # Special logic for the central node because its current values are predictions
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
    include_cal_lookback: bool = False,
    node_features: list = None,
    cal_columns: list = None,
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
    from graphsage_pyg import generate_node_features

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

    if include_cal_lookback:
        # Rolling calendar lookback window (unscaled exog already scaled by caller)
        cal_window   = x_scaled[:, exog_indices].copy()  # (lookback, cal_dim)
    else:
        cal_window   = None

    # Rolling calendar window for ts_seq construction (always maintained,
    # independent of include_cal_lookback).  rolling_cal[t] = calendar at
    # position t of the current lookback window.
    rolling_cal = x_scaled[:, exog_indices].copy() if cal_dim > 0 else None  # (lookback, cal_dim)

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
            )

            # ---- 2. Override node features with enhanced representations ----
            n_nodes = G_data.x.shape[0]
            x_new   = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)

            # Target node: full scaled lookback + [cal_lookback] + next-step calendar + stats
            cal_next = (future_exog[i] if future_exog.ndim > 1
                        else np.array([future_exog[i]], dtype=np.float32))
            cal_lb   = cal_window if include_cal_lookback else None
            
            selected_target = node_features 
            x_new[0] = torch.tensor(
                generate_node_features(target_window_scaled, cal_next=cal_next, cal_lookback=cal_lb,
                                       selected_features=selected_target, cal_columns=cal_columns),
                dtype=torch.float32,
            )

            # Neighbor nodes: pad to feature_dim
            for node_idx in range(1, n_nodes):
                orig_feat   = G_data.x[node_idx].numpy()
                neighbor_ts = orig_feat[:graph_window_size]
                
                selected_neighbor = node_features
                x_new[node_idx] = torch.tensor(
                    generate_node_features(neighbor_ts, selected_features=selected_neighbor,
                                           is_neighbor=True, pad_ts_to=lookback, cal_columns=cal_columns),
                    dtype=torch.float32,
                )

            G_data.x = x_new

            # ---- 3. Build ts_seq for LSTM ----
            # ts_seq[t] = (value_t, cal_{t+1}): at each lookback step the LSTM
            # sees the scaled target value and the calendar of the NEXT day.
            # The last step sees (value_{lookback-1}, cal_next), consistent with
            # the calendar already embedded in the target node features.
            if cal_dim > 0:
                _cal_shifted = np.vstack([
                    rolling_cal[1:],            # rows 1..lookback-1  → cal_{t+1}
                    cal_next.reshape(1, -1),    # row lookback-1      → cal_next
                ])  # (lookback, cal_dim)
                ts_seq_np = np.concatenate(
                    [target_window_scaled[:, None], _cal_shifted], axis=1
                )  # (lookback, 1+cal_dim)
            else:
                ts_seq_np = target_window_scaled[:, None]  # (lookback, 1)
            ts_seq = torch.from_numpy(ts_seq_np).float().unsqueeze(0).to(device)  # (1, L, 1+cal_dim)

            # ---- 4. Forward pass ----
            pyg_batch           = Batch.from_data_list([G_data]).to(device)
            target_node_indices = torch.tensor([0], dtype=torch.long, device=device)

            y_pred    = model(pyg_batch, target_node_indices, ts_seq)
            val_pred  = y_pred.item()

            unscaled_val = scaler.inverse_transform([[val_pred]])[0, 0]
            preds_unscaled.append(unscaled_val)

            # ---- 5. Shift rolling windows ----
            target_window_scaled = np.roll(target_window_scaled, -1)
            target_window_scaled[-1] = val_pred

            # Shift calendar windows
            if cal_dim > 0:
                rolling_cal = np.roll(rolling_cal, -1, axis=0)
                rolling_cal[-1] = cal_next
            if include_cal_lookback:
                cal_window = np.roll(cal_window, -1, axis=0)
                cal_window[-1] = cal_next

    return np.array(preds_unscaled).flatten()


def recursive_inference_mlp_no_graph(
    model: torch.nn.Module,
    scaler,
    recent_history: np.ndarray,
    future_exog: np.ndarray,
    target_channel: int = 0,
    device: Optional[str] = None,
    exog_cols: list = None,
    exog_scaler=None,
) -> np.ndarray:
    """
    Recursive 1-step-ahead inference for flat models (MLP or LSTM), no graph.
    When exog_cols and exog_scaler are provided, lag/rolling features are
    recomputed from model predictions at each step (avoids oracle leakage).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    horizon = len(future_exog)
    recent_history = np.asarray(recent_history, dtype=np.float32)
    if recent_history.ndim == 1:
        recent_history = recent_history[:, None]

    C_in = recent_history.shape[1]
    exog_indices = [idx for idx in range(C_in) if idx != target_channel]

    # Detect dynamic features (lags, rolling_mean) that need recursive update
    dynamic_features = []  # (col_idx_in_exog, feat_type, window_size)
    if exog_cols and exog_scaler is not None:
        for j, col in enumerate(exog_cols):
            try:
                if col.startswith("rolling_mean_excl_"):
                    dynamic_features.append((j, "rolling_mean_excl", int(col.split("_")[-1])))
                elif col.startswith("rolling_mean_"):
                    dynamic_features.append((j, "rolling_mean", int(col.split("_")[-1])))
                elif col.startswith("lag_"):
                    dynamic_features.append((j, "lag", int(col.split("_")[-1])))
            except ValueError:
                pass

    # Mutable copy so lag columns can be updated in-place
    future_exog = np.array(future_exog, dtype=np.float32).copy()

    # Unscaled demand history buffer for lag/rolling computation
    if dynamic_features:
        max_w = max(w for _, _, w in dynamic_features)
        unscaled_hist = recent_history[-max_w:, target_channel].tolist()
    else:
        unscaled_hist = []

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

            # Recursively update lag/rolling features for the next step
            if dynamic_features and (i + 1) < horizon:
                unscaled_hist.append(unscaled_val)
                next_exog_raw = exog_scaler.inverse_transform(
                    future_exog[i + 1].reshape(1, -1)
                )[0].copy()
                for j, feat_type, w in dynamic_features:
                    if feat_type == "lag":
                        next_exog_raw[j] = (
                            unscaled_hist[-w] if w <= len(unscaled_hist)
                            else (unscaled_hist[0] if unscaled_hist else 0.0)
                        )
                    elif feat_type in ("rolling_mean_excl", "rolling_mean"):
                        window_vals = unscaled_hist[-w:]
                        next_exog_raw[j] = np.mean(window_vals) if len(window_vals) > 0 else 0.0
                future_exog[i + 1] = exog_scaler.transform(next_exog_raw.reshape(1, -1))[0]

            input_window = np.roll(input_window, -1, axis=0)
            input_window[-1, target_channel] = val_pred
            if len(exog_indices) > 0 and (i + 1) < horizon:
                input_window[-1, exog_indices] = future_exog[i + 1]

    return np.array(preds_unscaled).flatten()

def recursive_inference_lstm_no_graph(model, test_start_idx, seq_length, 
                       val_scaled, exog_val_scaled, exog_test_scaled, exog_test,
                       scaler, exog_scaler, df_product, device, exog_cols,
                       forecast_window, seed, strategy, item_id, store_id, loss_type, script_dir):
    model.eval()
    
    current_seq = val_scaled[-seq_length:].tolist()
    current_exog_seq = []
    if exog_cols and len(exog_cols) > 0:
        current_exog_seq = exog_val_scaled[-seq_length + 1:].tolist() + [exog_test_scaled[0].tolist()]
    
    forecast = []
    
    dynamic_features = []
    if exog_cols:
        for idx, col in enumerate(exog_cols):
            try:
                if col.startswith("rolling_mean_excl_"):
                    dynamic_features.append((idx, "rolling_mean_excl", int(col.split("_")[-1])))
                elif col.startswith("rolling_mean_"):
                    dynamic_features.append((idx, "rolling_mean", int(col.split("_")[-1])))
                elif col.startswith("lag_"):
                    dynamic_features.append((idx, "lag", int(col.split("_")[-1])))
            except ValueError:
                pass
                    
    # Setup inference log file
    inf_log_dir = os.path.join(script_dir, f'inference_logs/seed_{seed}/{loss_type}/{strategy}')
    os.makedirs(inf_log_dir, exist_ok=True)
    inf_log_path = os.path.join(inf_log_dir, f'inference_item{item_id}_store{store_id}.csv')
                    
    start_inference_time = time.time()
    
    with open(inf_log_path, 'w') as inf_log_file:
        header_str = "Step,X,Predicted_Y_Scaled,Predicted_Y_Unscaled"
        if exog_cols and len(exog_cols) > 0:
            exog_cols_unscaled_str = ",".join([f"{col}_Unscaled" for col in exog_cols])
            exog_cols_scaled_str = ",".join([f"{col}_Scaled" for col in exog_cols])
            header_str += f",{exog_cols_unscaled_str},{exog_cols_scaled_str}"
        inf_log_file.write(header_str + "\n")
        
        with torch.no_grad():
            for step in range(forecast_window):
                if exog_cols and len(exog_cols) > 0:
                    current_seq_arr = np.array(current_seq).reshape(-1, 1)
                    current_exog_arr = np.array(current_exog_seq)
                    x_np = np.column_stack([current_seq_arr, current_exog_arr])
                else:
                    x_np = np.array(current_seq).reshape(-1, 1)

                x = torch.FloatTensor(x_np).unsqueeze(0).to(device)
                pred = model(x).cpu().numpy()[0, 0]
                forecast.append(pred)

                pred_unscaled = scaler.inverse_transform([[pred]])[0, 0]
                x_str = str(x_np.tolist()).replace('"', "'")
                
                if exog_cols and len(exog_cols) > 0:
                    last_exog_scaled = current_exog_arr[-1]
                    last_exog_raw = exog_scaler.inverse_transform(last_exog_scaled.reshape(1, -1))[0]
                    last_exog_unscaled_str = ",".join([str(v) for v in last_exog_raw.tolist()])
                    last_exog_scaled_str = ",".join([str(v) for v in last_exog_scaled.tolist()])
                    inf_log_file.write(f'{step},"{x_str}",{pred},{pred_unscaled},{last_exog_unscaled_str},{last_exog_scaled_str}\n')
                else:
                    inf_log_file.write(f'{step},"{x_str}",{pred},{pred_unscaled}\n')

                current_seq = current_seq[1:] + [pred]

                if exog_cols and len(exog_cols) > 0 and step + 1 < forecast_window:
                    next_exog_raw = exog_test[step + 1].copy()

                    if len(dynamic_features) > 0:
                        max_w = max([w for _, _, w in dynamic_features])
                        # current_seq[-1] is the prediction for 'step'. We are preparing exog for 'step + 1'
                        hist_unscaled = scaler.inverse_transform(np.array(current_seq[-max_w:]).reshape(-1, 1)).flatten()
                        
                        for idx, feat_type, w in dynamic_features:
                            if feat_type == "lag":
                                if w <= len(hist_unscaled):
                                    next_exog_raw[idx] = hist_unscaled[-w]
                                else:
                                    next_exog_raw[idx] = hist_unscaled[0] if len(hist_unscaled) > 0 else 0.0
                            elif feat_type == "rolling_mean_excl":
                                # For step+1, shift(1) means the window ends at step.
                                window_values = hist_unscaled[-w:]
                                next_exog_raw[idx] = np.mean(window_values) if len(window_values) > 0 else 0.0
                            elif feat_type == "rolling_mean":
                                # For step+1, rolling included the future target at step+1, which we don't have.
                                # Approximating by averaging the available window up to step.
                                window_values = hist_unscaled[-w:]
                                next_exog_raw[idx] = np.mean(window_values) if len(window_values) > 0 else 0.0

                    next_exog_scaled = exog_scaler.transform(next_exog_raw.reshape(1, -1))[0]
                    current_exog_seq = current_exog_seq[1:] + [next_exog_scaled.tolist()]
                
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    inference_time = time.time() - start_inference_time
    
    return forecast, inference_time


# Alias: clearer name for the GraphSAGE+LSTM inference function
recursive_inference_sage_lstm = recursive_inference_pure_sage