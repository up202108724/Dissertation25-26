"""
utils.py — Shared utilities for StaticGCN_LSTM.
Mirrors GNN/FixedGraphs/SimpleGNN/LSTM/utils.py (exog feature generation only).
"""

import pandas as pd
import numpy as np
import holidays
import torch

def generate_exogenous_features(df, exog_cols, date_col='date', target_col='value', group_cols=None):
    """
    Generates specific calendar, cyclical, and holiday exogenous features for a DataFrame
    based on the provided `exog_cols` list.
    """
    df = df.copy()

    if group_cols is None:
        group_cols = [c for c in ['item_id', 'store_id'] if c in df.columns]
        if not group_cols:
            group_cols = None

    def _get_holidays():
        us_holidays = holidays.US()
        return us_holidays, pd.to_datetime(sorted(us_holidays.keys()))

    _holidays_cache = None

    def get_holiday_dates():
        nonlocal _holidays_cache
        if _holidays_cache is None:
            _holidays_cache = _get_holidays()
        return _holidays_cache

    builders = {
        "day_of_week":    lambda d: d[date_col].dt.dayofweek.astype(int),
        "day_of_month":   lambda d: d[date_col].dt.day.astype(int),
        "month":          lambda d: d[date_col].dt.month.astype(int),
        "quarter":        lambda d: d[date_col].dt.quarter.astype(int),
        "week_of_year":   lambda d: d[date_col].dt.isocalendar().week.astype(int),
        "week_of_month":  lambda d: ((d[date_col].dt.day - 1) // 7 + 1).astype(int),
        "is_weekend":     lambda d: d[date_col].dt.dayofweek.isin([5, 6]).astype(int),
        "is_monday":      lambda d: (d[date_col].dt.dayofweek == 0).astype(int),
        "is_friday":      lambda d: (d[date_col].dt.dayofweek == 4).astype(int),
        "is_month_start": lambda d: d[date_col].dt.is_month_start.astype(int),
        "is_month_end":   lambda d: d[date_col].dt.is_month_end.astype(int),
        "is_quarter_start": lambda d: d[date_col].dt.is_quarter_start.astype(int),
        "is_quarter_end":   lambda d: d[date_col].dt.is_quarter_end.astype(int),
        "dow_sin":  lambda d: np.sin(2 * np.pi * d[date_col].dt.dayofweek / 7),
        "dow_cos":  lambda d: np.cos(2 * np.pi * d[date_col].dt.dayofweek / 7),
        "doy_sin":  lambda d: np.sin(2 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),
        "doy_cos":  lambda d: np.cos(2 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),
        "month_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.month - 1) / 12.0),
        "month_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.month - 1) / 12.0),
        "woy_sin":  lambda d: np.sin(2 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),
        "woy_cos":  lambda d: np.cos(2 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),
        "is_holiday": lambda d: d[date_col].isin(get_holiday_dates()[1]).astype(int),
        "is_thanksgiving": lambda d: d[date_col].apply(
            lambda x: 1 if get_holiday_dates()[0].get(x) == "Thanksgiving Day" else 0
        ),
        "is_black_friday": lambda d: d[date_col].isin(
            pd.to_datetime([day for day, name in get_holiday_dates()[0].items()
                            if name == "Thanksgiving Day"]) + pd.Timedelta(days=1)
        ).astype(int),
        "is_christmas":     lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 25)).astype(int),
        "is_christmas_eve": lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 24)).astype(int),
        "is_new_year_eve":  lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 31)).astype(int),
        "is_bridge_day": lambda d: pd.Series(0, index=d.index),
        "rolling_mean_excl_7": lambda d: (
            d.groupby(group_cols)[target_col].transform(
                lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
            ).fillna(0) if group_cols else d[target_col].shift(1).rolling(7, min_periods=1).mean().fillna(0)
        ),
    }

    for col in exog_cols:
        if col in builders:
            df[col] = builders[col](df)
        elif col.startswith("is_pre_holiday_") or col.startswith("is_post_holiday_"):
            parts   = col.split("_")
            lag     = int(parts[-1])
            is_pre  = "pre" in parts
            _, holiday_dates = get_holiday_dates()
            df[col] = 0
            for h in holiday_dates:
                target_date = h - pd.Timedelta(days=lag) if is_pre else h + pd.Timedelta(days=lag)
                df.loc[df[date_col] == target_date, col] = 1
        elif col.startswith("lag_"):
            lag = int(col.split("_")[-1])
            if group_cols:
                df[col] = df.groupby(group_cols)[target_col].shift(lag).fillna(0)
            else:
                df[col] = df[target_col].shift(lag).fillna(0)
        elif col.startswith("rolling_mean_excl_"):
            window = int(col.split("_")[-1])
            if group_cols:
                df[col] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                ).fillna(0)
            else:
                df[col] = df[target_col].shift(1).rolling(window, min_periods=1).mean().fillna(0)

    return df

def compute_similarities_1vsAll(target_ts, all_ts, metric='pearson', eps=1e-12):
    """
    Computes similarities between target_ts (1D) and all_ts (2D) using PyTorch.
    Optimized for 1-vs-all.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.tensor(target_ts, dtype=torch.float32, device=device).unsqueeze(0)
    X = torch.tensor(all_ts, dtype=torch.float32, device=device)
    
    if metric == 'pearson':
        target_mean = torch.mean(target, dim=1, keepdim=True)
        X_mean = torch.mean(X, dim=1, keepdim=True)
        
        target_centered = target - target_mean
        X_centered = X - X_mean
        
        cov = torch.sum(target_centered * X_centered, dim=1)
        target_var = torch.sqrt(torch.sum(target_centered**2, dim=1))
        X_var = torch.sqrt(torch.sum(X_centered**2, dim=1))
        
        sim = cov / (target_var * X_var + eps)
        return sim.cpu().numpy()
        
    elif metric == 'spearman':
        _, target_indices = torch.sort(target, dim=1)
        target_ranks = torch.empty_like(target)
        target_ranks.scatter_(1, target_indices, torch.arange(1, target.shape[1]+1, dtype=torch.float32, device=device).unsqueeze(0))
        
        _, X_indices = torch.sort(X, dim=1)
        X_ranks = torch.empty_like(X)
        X_ranks.scatter_(1, X_indices, torch.arange(1, X.shape[1]+1, dtype=torch.float32, device=device).unsqueeze(0).expand_as(X))
        
        target_mean = torch.mean(target_ranks, dim=1, keepdim=True)
        X_mean = torch.mean(X_ranks, dim=1, keepdim=True)
        
        target_centered = target_ranks - target_mean
        X_centered = X_ranks - X_mean
        
        cov = torch.sum(target_centered * X_centered, dim=1)
        target_var = torch.sqrt(torch.sum(target_centered**2, dim=1))
        X_var = torch.sqrt(torch.sum(X_centered**2, dim=1))
        
        sim = cov / (target_var * X_var + eps)
        return sim.cpu().numpy()
        
    elif metric == 'kendall':
        seq_len = target.shape[1]
        if seq_len < 2:
            return torch.ones(X.shape[0], device=device).cpu().numpy()
            
        idx1, idx2 = torch.triu_indices(seq_len, seq_len, offset=1, device=device)
        
        target_diffs = target[:, idx1] - target[:, idx2]
        target_signs = torch.sign(target_diffs)
        
        X_diffs = X[:, idx1] - X[:, idx2]
        X_signs = torch.sign(X_diffs)
        
        S = torch.sum(target_signs * X_signs, dim=1)
        
        target_non_ties = torch.sum(target_signs**2, dim=1)
        X_non_ties = torch.sum(X_signs**2, dim=1)
        
        denom = torch.sqrt(target_non_ties * X_non_ties)
        
        sim = torch.where(denom == 0, torch.tensor(0.0, device=device), S / denom)
        return sim.cpu().numpy()
        
    else:
        raise ValueError(f"Metric {metric} not supported")


def neighbourhood_graph(product_id, df, metric, metric_type, window_size, compute_func, 
                        threshold=None, percentile=None, step_size=1, cat_labels=None, plot_dir=None, residuals=False,
                        enable_edges_within_star=True, enable_second_degree=False, num_plots=None, train_end_idx=None):
    """
    Constructs a graph by iterating over sliding time windows of the time series data.
    Finds items within the specified metric thresholds or percentiles to product_id.
    Returns PyTorch Geometric Data objects instead of NetworkX graphs.
    
    Args:
        product_id: ID of the central product
        df: DataFrame with time series (rows=items, cols=time steps)
        metric: Metric string (e.g. 'euclidean', 'pearson')
        metric_type: 'distance' or 'similarity'
        window_size: Size of the sliding window
        compute_func: Function to compute 1-vs-All metric. Must take (target_ts, all_ts, metric)
        threshold: Strict cutoff threshold
        percentile: Top % of closest/most similar items
        step_size: Number of timesteps the window slides forward
        cat_labels: Dictionary mapping item IDs to category labels
        plot_dir: Save HTML plots in this directory
    """
    from torch_geometric.data import Data

    if threshold is None and percentile is None:
        raise ValueError("Must provide a threshold or percentile.")
        
    if product_id not in df.index:
        raise ValueError(f"Product ID {product_id} not found in DataFrame index.")
        
    time_steps = df.shape[1]
    
    # --- PHASE 1: Pre-calculate a global threshold across ALL windows ---
    global_threshold = threshold
    if threshold is None:
        all_valid_vals = []
        scan_end_limit = time_steps
        if train_end_idx is not None:
            scan_end_limit = min(scan_end_limit, train_end_idx)
            
        for start_idx in range(0, scan_end_limit - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_data = df.iloc[:, start_idx:end_idx]
            
            target_ts = window_data.loc[product_id].values
            if np.sum(np.abs(target_ts)) == 0:
                continue
                
            all_ts = window_data.values
            item_ids = window_data.index.values
            
            vals = compute_func(target_ts, all_ts, metric=metric)
            
            active_items_mask = np.sum(np.abs(all_ts), axis=1) > 0
            valid_mask = (item_ids != product_id) & active_items_mask
            all_valid_vals.extend(vals[valid_mask])
            
        all_valid_vals = np.array(all_valid_vals)
        all_valid_vals = all_valid_vals[np.isfinite(all_valid_vals)]
        
        if len(all_valid_vals) > 0:
            if percentile is not None:
                # Calculate exactly how many items correspond to the top-k%
                k = max(1, int(len(all_valid_vals) * (percentile / 100.0)))
                
                if metric_type == 'distance':
                    # Sort distances ascending (smaller is better). Slice top k, take the max.
                    top_k_vals = np.sort(all_valid_vals)[:k]
                    global_threshold = top_k_vals[-1]
                else: # similarity
                    # Sort similarities descending (larger is better). Slice top k, take the min.
                    top_k_vals = np.sort(all_valid_vals)[::-1][:k]
                    global_threshold = top_k_vals[-1]
        else:
            global_threshold = 0.0
            
    # --- PHASE 2: Build PyG Graphs using the single global_threshold ---
    graphs = []
    
    windows_range = list(range(0, time_steps - window_size + 1, step_size))
    total_graphs = len(windows_range)
    print(f"Building {total_graphs} sliding window PyG graphs...")
    
    for i, start_idx in enumerate(windows_range):
        if (i + 1) % 50 == 0 or (i + 1) == total_graphs:
            print(f"  Built {i + 1}/{total_graphs} graphs...", end='\r' if (i + 1) < total_graphs else '\n')
            
        end_idx = start_idx + window_size
        window_data = df.iloc[:, start_idx:end_idx]
        
        target_ts = window_data.loc[product_id].values
        all_ts = window_data.values
        item_ids = window_data.index.values
        
        vals = compute_func(target_ts, all_ts, metric=metric)
        
        active_items_mask = np.sum(np.abs(all_ts), axis=1) > 0
        valid_mask = (item_ids != product_id) & active_items_mask
        
        if np.sum(np.abs(target_ts)) == 0:
            valid_mask[:] = False
            
        valid_item_ids = item_ids[valid_mask]
        valid_vals = vals[valid_mask]
        valid_original_idxs = np.arange(len(item_ids))[valid_mask]
        
        if len(valid_vals) == 0:
            mask = np.array([], dtype=bool)
            current_threshold = 0
        else:
            if metric_type == 'distance':
                mask = valid_vals <= global_threshold
                current_threshold = global_threshold
            else: # similarity
                mask = valid_vals >= global_threshold
                current_threshold = global_threshold

        selected_vals = valid_vals[mask]
        selected_ids = valid_item_ids[mask]
        selected_orig_idxs = valid_original_idxs[mask]

        # Garantir ordem determinística
        sort_idx = np.argsort(selected_ids)
        selected_vals = selected_vals[sort_idx]
        selected_ids = selected_ids[sort_idx]
        selected_orig_idxs = selected_orig_idxs[sort_idx]

        # PyG Graph mapping setup
        graph_nodes = [product_id] + selected_ids.tolist()
        node_to_idx = {n_id: idx for idx, n_id in enumerate(graph_nodes)}
        
        edge_list = []
        edge_weights = []

        neighbor_indices = selected_orig_idxs.tolist()
        
        # Edges from central node to neighbors
        for val, other_id in zip(selected_vals, selected_ids):
            u = 0  # product_id is always 0
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
                vals_sub = compute_func(target_neighbor_ts, all_neighbors_ts, metric=metric)
                
                for j, (idx2, val_sub) in enumerate(zip(neighbor_indices, vals_sub)):
                    if i < j:
                        if metric_type == 'distance':
                            if val_sub <= current_threshold:
                                edge_weight = 1.0 / (1.0 + float(val_sub))
                                u, v = node_to_idx[item_ids[idx1]], node_to_idx[item_ids[idx2]]
                                edge_list.append([u, v])
                                edge_list.append([v, u])
                                edge_weights.extend([edge_weight, edge_weight])
                        else:
                            if val_sub >= current_threshold:
                                edge_weight = max(0.0, float(val_sub))
                                u, v = node_to_idx[item_ids[idx1]], node_to_idx[item_ids[idx2]]
                                edge_list.append([u, v])
                                edge_list.append([v, u])
                                edge_weights.extend([edge_weight, edge_weight])

        if enable_second_degree and len(neighbor_indices) > 0:
            for idx1 in neighbor_indices:
                target_neighbor_ts = all_ts[idx1] 
                vals_sub = compute_func(target_neighbor_ts, all_ts, metric=metric)
                
                for valid_idx, is_valid in enumerate(valid_mask):
                    if is_valid and valid_idx != idx1:
                        val_sub = vals_sub[valid_idx]
                        other_id = item_ids[valid_idx]
                        
                        add_edge = False
                        if metric_type == 'distance':
                            if val_sub <= current_threshold:
                                add_edge = True
                                edge_weight = 1.0 / (1.0 + float(val_sub)) 
                        else:
                            if val_sub >= current_threshold:
                                add_edge = True
                                edge_weight = max(0.0, float(val_sub))
                                
                        if add_edge:
                            if other_id not in node_to_idx:
                                node_to_idx[other_id] = len(graph_nodes)
                                graph_nodes.append(other_id)
                                
                            u, v = node_to_idx[item_ids[idx1]], node_to_idx[other_id]
                            # check if edge already exists, but for simplicity we can just add
                            # assuming it will handle undirected properly with redundant checks
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
            # Locate node row in the window
            row = window_data.loc[n_id].values
            features = compute_node_features(row)
            x_matrix.append(features)
            
        x = torch.tensor(np.array(x_matrix), dtype=torch.float)

        # PyG Data Object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.central_node_idx = 0
        data.target_id = product_id
        
        start_date = str(window_data.columns[0]).split(' ')[0].split('T')[0]
        end_date = str(window_data.columns[-1]).split(' ')[0].split('T')[0]
        data.start_date = start_date
        data.end_date = end_date

        graphs.append(data)
                
    return graphs, global_threshold