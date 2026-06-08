import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
import os
import seaborn as sns
import sys
from tslearn.metrics import cdist_dtw
import torch
from tslearn.metrics import dtw
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)                                          # local modules (inference, train, ...)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '../..')))  # DynamicSimilarities/ (plots, utils)

from plots import plot_networkx_plotly
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
            
    # --- PHASE 2: Build Graphs using the single global_threshold ---
    graphs = []
    
    windows_range = list(range(0, time_steps - window_size + 1, step_size))
    total_graphs = len(windows_range)
    print(f"Building {total_graphs} sliding window graphs...")
    
    for i, start_idx in enumerate(windows_range):
        if (i + 1) % 50 == 0 or (i + 1) == total_graphs:
            print(f"  Built {i + 1}/{total_graphs} graphs...", end='\r' if (i + 1) < total_graphs else '\n')
            
        G = nx.Graph()
        
        cat = cat_labels.get(product_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
        G.add_node(product_id, cat_label=cat)
        
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

        # Garantir ordem determinística (fixar matriz de adjacências independentemente da query ou extração)
        sort_idx = np.argsort(selected_ids)
        selected_vals = selected_vals[sort_idx]
        selected_ids = selected_ids[sort_idx]
        selected_orig_idxs = selected_orig_idxs[sort_idx]

        neighbor_indices = []
        for orig_idx, val, other_id in zip(selected_orig_idxs, selected_vals, selected_ids):
            cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
            G.add_node(other_id, cat_label=cat_other)
            G.add_edge(product_id, other_id, weight=float(val))
            neighbor_indices.append(orig_idx)
            
        if enable_edges_within_star and len(neighbor_indices) > 1:
            all_neighbors_ts = all_ts[neighbor_indices]
            for i, idx1 in enumerate(neighbor_indices):
                target_neighbor_ts = all_ts[idx1] 
                vals_sub = compute_func(target_neighbor_ts, all_neighbors_ts, metric=metric)
                
                for j, (idx2, val_sub) in enumerate(zip(neighbor_indices, vals_sub)):
                    if i < j:
                        if metric_type == 'distance':
                            if val_sub <= current_threshold:
                                edge_weight = float(val_sub)
                                G.add_edge(item_ids[idx1], item_ids[idx2], weight=edge_weight)
                        else:
                            if val_sub >= current_threshold:
                                edge_weight = max(0.0, float(val_sub))
                                G.add_edge(item_ids[idx1], item_ids[idx2], weight=edge_weight)

        if enable_second_degree and len(neighbor_indices) > 0:
            for idx1 in neighbor_indices:
                target_neighbor_ts = all_ts[idx1] 
                vals_sub = compute_func(target_neighbor_ts, all_ts, metric=metric)
                
                for valid_idx, is_valid in enumerate(valid_mask):
                    if is_valid and valid_idx != idx1:  # Must be active and not itself nor the central product
                        val_sub = vals_sub[valid_idx]
                        other_id = item_ids[valid_idx]
                        
                        add_edge = False
                        if metric_type == 'distance':
                            if val_sub <= current_threshold:
                                add_edge = True
                                edge_weight = float(val_sub)
                        else:
                            if val_sub >= current_threshold:
                                add_edge = True
                                edge_weight = max(0.0, float(val_sub))
                                
                        if add_edge:
                            if not G.has_node(other_id):
                                cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
                                G.add_node(other_id, cat_label=cat_other)
                            if not G.has_edge(item_ids[idx1], other_id):
                                G.add_edge(item_ids[idx1], other_id, weight=edge_weight)

        start_date = str(window_data.columns[0]).split(' ')[0].split('T')[0]
        end_date = str(window_data.columns[-1]).split(' ')[0].split('T')[0]
        
        G.graph['start_date'] = start_date
        G.graph['end_date'] = end_date

        graphs.append(G)
                
    return graphs, global_threshold
def compute_distances_1vsAll(target_ts, all_ts, metric='amplitude_offset', eps=1e-12, normalize_inputs=False):
    """
    Computes distances between target_ts (1D) and all_ts (2D) using PyTorch.
    Optimized for 1-vs-all.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.tensor(target_ts, dtype=torch.float32, device=device).unsqueeze(0)
    X = torch.tensor(all_ts, dtype=torch.float32, device=device)
    '''
    if normalize_inputs and metric in ['cid', 'dtw']:
        target_mean = torch.mean(target, dim=1, keepdim=True)
        target_std = torch.std(target, dim=1, keepdim=True) + eps
        target = (target - target_mean) / target_std
        
        X_mean = torch.mean(X, dim=1, keepdim=True)
        X_std = torch.std(X, dim=1, keepdim=True) + eps
        X = (X - X_mean) / X_std
    '''
    if metric == 'manhattan':
        # sum(|x - y|)
        dist = torch.sum(torch.abs(X - target), dim=1)
        return dist.cpu().numpy()
    
    elif metric == 'hamming':
        target_bin = (target > 0).float()
        X_bin = (X > 0).float()
        diffs = torch.abs(X_bin - target_bin)
        dist = torch.mean(diffs, dim=1)
        return dist.cpu().numpy()
    
    elif metric == 'amplitude_offset':
        target_mean = torch.mean(target, dim=1, keepdim=True)
        target_std = torch.std(target, dim=1, keepdim=True) + eps
        target_norm = (target - target_mean) / target_std
        
        X_mean = torch.mean(X, dim=1, keepdim=True)
        X_std = torch.std(X, dim=1, keepdim=True) + eps
        X_norm = (X - X_mean) / X_std
        
        dist = torch.cdist(target_norm, X_norm, p=2).squeeze(0)
        return dist.cpu().numpy()
        
    elif metric == 'slope_consistency':
        target_min = torch.min(target, dim=1, keepdim=True)[0]
        target_max = torch.max(target, dim=1, keepdim=True)[0]
        target_norm = (target - target_min) / (target_max - target_min + eps)
        
        X_min = torch.min(X, dim=1, keepdim=True)[0]
        X_max = torch.max(X, dim=1, keepdim=True)[0]
        X_norm = (X - X_min) / (X_max - X_min + eps)
        
        diffs = X_norm - target_norm
        dist = torch.var(diffs, dim=1, unbiased=False)
        return dist.cpu().numpy()
        
    elif metric == 'cid':
        ed_dist = torch.cdist(target, X, p=2).squeeze(0)
        
        # Complexity estimation
        ce_target = torch.sqrt(torch.sum(torch.diff(target, dim=1) ** 2, dim=1))
        ce_target_safe = torch.maximum(ce_target, torch.tensor(eps, device=device))
        
        ce_X = torch.sqrt(torch.sum(torch.diff(X, dim=1) ** 2, dim=1))
        ce_X_safe = torch.maximum(ce_X, torch.tensor(eps, device=device))
        
        ce_max = torch.maximum(ce_target_safe, ce_X_safe)
        ce_min = torch.minimum(ce_target_safe, ce_X_safe)
        cf_dist = ce_max / ce_min
        
        cid_dist = ed_dist * cf_dist
        return cid_dist.cpu().numpy()
        
    elif metric == 'dtw':
        # Use tslearn's optimized cdist_dtw for 1-vs-all vectorization 
        X_np = X.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # For a 15-point window, a Sakoe-Chiba radius of 2 or 3 (roughly ~10-20% of length) is optimal.
        # This constrains the pathological warping and drastically speeds up the calculation.
        dist = cdist_dtw(
            target_np.reshape(1, -1), 
            X_np, 
            global_constraint="sakoe_chiba", 
            sakoe_chiba_radius=2, 
            n_jobs=-1
        )
        
        # cdist_dtw returns a 2D distance matrix of shape (1, n_X), so we flatten it
        return dist.flatten()
        
    elif metric == 'phase_invariance':
        # Phase Invariance (Circular Shift) Euclidean Match
        # Find the minimum Euclidean distance across all possible circular shifts of the sequences
        X_np = X.cpu().numpy()
        target_np = target.cpu().numpy()
        
        n_X = X_np.shape[0]
        seq_len = X_np.shape[1]
        
        min_dists = np.full(n_X, np.inf)
        
        for shift in range(seq_len):
            shifted_X = np.roll(X_np, shift, axis=1)
            # Compute euclidean distance for this specific shift broadcasted across all rows
            current_dists = np.sqrt(np.sum((shifted_X - target_np)**2, axis=1))
            min_dists = np.minimum(min_dists, current_dists)
            
        return min_dists
        
    elif metric == 'lorentzian':
        # Lorentzian distance: sum(ln(1 + |x - y|))
        # Robust against outliers and noise
        diffs = torch.abs(X - target)
        dist = torch.sum(torch.log1p(diffs), dim=1)
        return dist.cpu().numpy()
    
    elif metric == 'twed':
        try:
            from sktime.distances import twe_distance
            X_np = X.cpu().numpy()
            target_np = target.cpu().numpy().flatten()
            # nu: stiffness, lmbda: penalty for deletion/insertion
            return np.array([twe_distance(target_np, x, nu=0.001, lmbda=1.0) for x in X_np])
        except ImportError as e:
            raise ValueError(f"metric='twed' failed to import: {e}")
        
    elif metric == 'erp':
        try:
            from sktime.distances import erp_distance
            X_np = X.cpu().numpy()
            target_np = target.cpu().numpy().flatten()
            # g is the gap value (usually 0.0 for Z-normalized data)
            return np.array([erp_distance(target_np, x, g=0.0) for x in X_np])
        except ImportError:
            raise ValueError("metric='erp' requires 'sktime'.")
        
    elif metric == 'stid':
        # Simplified STID: find min Euclidean distance across shifts after local scaling
        X_np = X.cpu().numpy()
        target_np = target.cpu().numpy()
        n_X, seq_len = X_np.shape
        min_dists = np.full(n_X, np.inf)
        
        # Search across possible shifts (w)
        for shift in range(-2, 3): # Local search window
            shifted_X = np.roll(X_np, shift, axis=1)
            # Optimal scaling alpha for each pair (simplified)
            # alpha = (X.T @ Y) / ||Y||^2
            dot_product = np.sum(shifted_X * target_np, axis=1)
            target_norm_sq = np.sum(target_np**2) + eps
            alpha = dot_product / target_norm_sq
            
            current_dists = np.linalg.norm(shifted_X - (alpha[:, None] * target_np), axis=1)
            min_dists = np.minimum(min_dists, current_dists)
            
        return min_dists
    
    elif metric == 'sbd':
        # Shape-Based Distance (SBD) using cross-correlation via FFT
        # SBD = 1 - max(NCC)
        X_np = X.cpu().numpy()
        target_np = target.cpu().numpy().flatten()
        
        n_X, seq_len = X_np.shape
        dists = np.zeros(n_X)
        
        target_norm = np.linalg.norm(target_np)
        target_norm = max(target_norm, eps)
        
        for i in range(n_X):
            x = X_np[i]
            x_norm = np.linalg.norm(x)
            x_norm = max(x_norm, eps)
            
            # Cross-correlation using scipy/numpy
            pad_len = 2 * seq_len - 1
            fft_x = np.fft.fft(x, n=pad_len)
            fft_y = np.fft.fft(target_np[::-1], n=pad_len)
            cc = np.real(np.fft.ifft(fft_x * fft_y))
            
            ncc = np.max(cc) / (target_norm * x_norm)
            # SBD falls in [0, 2]
            dists[i] = 1 - ncc
            
        return dists
        
    elif metric == 'msm':
        # Move-Split-Merge (MSM) is a metric elastic distance.
        try:
            from sktime.distances import msm_distance
            X_np = X.cpu().numpy()
            target_np = target.cpu().numpy().flatten()
            return np.array([msm_distance(target_np, x) for x in X_np])
        except ImportError:
            raise ValueError("metric='msm' requires 'sktime' to be installed. Run `pip install sktime`.")
            
    elif metric == 'edr' or metric == 'lcss':
        # Edit Distance on Real sequence (EDR) / Longest Common Subsequence (LCSS)
        X_np = X.cpu().numpy()
        target_np = target.cpu().numpy().flatten()
        
        if metric == 'lcss':
            try:
                from tslearn.metrics import lcss
                # Note: LCSS returns similarity ([0, 1]), so distance is 1 - similarity
                epsilon = 0.5 # A common threshold relative to scale; can be tuned
                return np.array([1 - lcss(target_np, x, eps=epsilon) for x in X_np])
            except ImportError:
                raise ValueError("metric='lcss' requires 'tslearn' to be installed.")
        else:
            try:
                from sktime.distances import edr_distance
                # EDR requires an epsilon threshold for what counts as a match
                epsilon = 0.5 
                return np.array([edr_distance(target_np, x, epsilon=epsilon) for x in X_np])
            except ImportError:
                raise ValueError("metric='edr' requires 'sktime' to be installed. Run `pip install sktime`.")
        
    else:
        raise ValueError(f"Metric {metric} not supported")
# Exogenous feature utils
def scale_exogenous_features(df, train_slice, categorical_cols=None, continuous_cols=None, binary_cols=None):
    df_scaled = df.copy()
    final_exog_cols = []
    
    cat_scaler = None
    cont_scaler = None
    
    # Scale categorical columns with OneHotEncoder
    if categorical_cols:
        cat_train = df.iloc[train_slice][categorical_cols].values
        cat_scaler = MinMaxScaler()  # Use MinMaxScaler to convert categories to a 0-1 range before OHE
        cat_scaler.fit(cat_train)
        
        cat_full_scaled = cat_scaler.transform(df[categorical_cols].values)
        cat_feature_names = cat_scaler.get_feature_names_out(categorical_cols)
        
        # Add new OHE columns and drop the originals
        df_cat = pd.DataFrame(cat_full_scaled, columns=cat_feature_names, index=df.index)
        df_scaled = pd.concat([df_scaled.drop(columns=categorical_cols), df_cat], axis=1)
        final_exog_cols.extend(cat_feature_names)
        
    # Scale continuous columns
    if continuous_cols:
        cont_train = df.iloc[train_slice][continuous_cols].values
        cont_scaler = MinMaxScaler()
        cont_scaler.fit(cont_train)
        
        cont_full_scaled = cont_scaler.transform(df[continuous_cols].values)
        df_scaled[continuous_cols] = cont_full_scaled
        final_exog_cols.extend(continuous_cols)
        
    # Keep binary columns as they are
    if binary_cols:
        final_exog_cols.extend(binary_cols)
        
    return df_scaled, final_exog_cols, cat_scaler, cont_scaler

def build_dynamic_exog_row(next_date, target_history, exog_cols,
                           lag_cols=None, rolling_cols=None, df_row=None):
    """
    Build one RAW exogenous row for next_date using current RAW target history.
    """
    row = {}

    if lag_cols:
        for col in lag_cols:
            lag = int(col.split('_')[1])
            row[col] = target_history[-lag] if len(target_history) >= lag else 0.0

    if rolling_cols:
        for col in rolling_cols:
            window = int(col.split('_')[2])
            if len(target_history) >= window:
                row[col] = float(np.mean(target_history[-window:]))
            else:
                row[col] = float(np.mean(target_history)) if len(target_history) > 0 else 0.0

    if df_row is not None:
        for col in exog_cols:
            if col not in row and col in df_row.index:
                row[col] = df_row[col]

    return pd.DataFrame([[row.get(col, 0.0) for col in exog_cols]], columns=exog_cols)



def transform_exog_row(row_df, categorical_cols=None, continuous_cols=None, binary_cols=None,
                       cat_scaler=None, cont_scaler=None, final_exog_cols=None):
    parts = []

    if categorical_cols:
        # Force column order
        cat_arr = cat_scaler.transform(row_df[categorical_cols].values)
        cat_names = cat_scaler.get_feature_names_out(categorical_cols)
        parts.append(pd.DataFrame(cat_arr, columns=cat_names, index=row_df.index))

    if continuous_cols:
        # Force column order precisely as fitted
        cont_arr = cont_scaler.transform(row_df[continuous_cols].values)
        parts.append(pd.DataFrame(cont_arr, columns=continuous_cols, index=row_df.index))

    if binary_cols:
        # Force column order
        parts.append(row_df[binary_cols].copy())

    out = pd.concat(parts, axis=1)

    if final_exog_cols is not None:
        # Reindex to ensure strictly ordered columns matching the training tensors
        out = out.reindex(columns=final_exog_cols, fill_value=0.0)

    return out


def plot_dynamic_graphs(graphs, product_id, metric, plot_dir, residuals=False, enable_edges_within_star=True, enable_second_degree=False, num_plots=None, window_size=None, step_size=None, threshold=None, percentile=None):
    if plot_dir is None:
        return
        
    import random
    import os
    # Filter valid graphs first
    valid_graphs = [g for g in graphs if len(g.nodes) > 1]
    
    # Decide which ones to plot
    plots_to_draw = valid_graphs
    if num_plots is not None and num_plots < len(valid_graphs):
        plots_to_draw = random.sample(valid_graphs, num_plots)
        
    for G_to_plot in plots_to_draw:    
        current_plot_dir = plot_dir
        if residuals:
            current_plot_dir = os.path.join(plot_dir, "residuals")
        os.makedirs(current_plot_dir, exist_ok=True)
        
        plot_prefix = "residual_" if residuals else ""
        if not enable_edges_within_star:
            plot_prefix += "star_"
        if enable_second_degree:
            plot_prefix += "2nd_degree_"
            
        start_d = G_to_plot.graph.get('start_date', 'Unknown')
        end_d = G_to_plot.graph.get('end_date', 'Unknown')
            
        plot_path = os.path.join(current_plot_dir, f'{plot_prefix}graph_{product_id}_{start_d}_to_{end_d}.html')
        try:
            
            print(f"Saving plot to {plot_path} with {len(G_to_plot.nodes)} nodes and {len(G_to_plot.edges)} edges...")
            
            # Construct comprehensive title
            details = []
            details.append(f"Metric: {metric}")
            if window_size is not None: details.append(f"Window: {window_size}")
            if step_size is not None: details.append(f"Step: {step_size}")
            if threshold is not None: details.append(f"Thresh: {threshold}")
            if percentile is not None: details.append(f"Pct: {percentile}")
            details.append(f"Edges inside Star: {enable_edges_within_star}")
            details.append(f"2nd Degree: {enable_second_degree}")
            
            title_str = f"Neighbors of Product {product_id}<br>{' | '.join(details)}<br>Date: {start_d} to {end_d}"
            
            plot_networkx_plotly(
                G_to_plot, 
                title=title_str,
                save_path=plot_path,
                target_node=product_id
            )
        except Exception as e:
            print(f"Plotting skipped due to error: {e}")

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def compute_metrics(y_test, y_pred):
    
    def POCID(y_test, y_pred):
        diff_original = y_test[1:] - y_test[:-1]
        diff_pred = y_pred[1:] - y_pred[:-1]
        is_positive = (diff_original * diff_pred) > 0
        return is_positive.sum() / len(is_positive) if len(is_positive) > 0 else 0.0

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    bias = np.mean(y_pred - y_test)
    score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)
    pocid = POCID(y_test, y_pred)
    return {"rmse": rmse, "mae": mae, "bias": bias, "score": score, "pocid": pocid}

def analyze_distance_distribution(df, product_id, window_size, metrics, plot_dir):
    """
    Computes pair-wise distances across all valid windows for the target product
    and plots a histogram to help determine percentiles and thresholds.
    """
    if product_id not in df.index:
        raise ValueError(f"Product ID {product_id} not found in DataFrame index.")
        
    print(f"\n--- Analyzing Distance Distributions for {product_id} across ALL windows ---")
    
    time_steps = df.shape[1]
    os.makedirs(plot_dir, exist_ok=True)
    
    aggregated_dists = {metric: [] for metric in metrics}
    
    # Iterate over all windows
    for start_idx in range(0, time_steps - window_size + 1):
        end_idx = start_idx + window_size
        window_data = df.iloc[:, start_idx:end_idx]
        target_ts = window_data.loc[product_id].values
        
        # Skip if target product has 0 sales in this window (as it would produce meaningless relations)
        if np.sum(np.abs(target_ts)) == 0:
            continue
            
        all_ts = window_data.values
        item_ids = window_data.index.values
        
        # Filter active items exactly like in graph construction
        active_items_mask = np.sum(np.abs(all_ts), axis=1) > 0
        valid_mask = (item_ids != product_id) & active_items_mask
        valid_all_ts = all_ts[valid_mask]
        
        for metric in metrics:
            try:
                # Compute distances from target to all other valid items in this window
                dists = compute_distances_1vsAll(target_ts, valid_all_ts, metric=metric)
                valid_dists = dists[np.isfinite(dists) & ~np.isnan(dists)]
                if len(valid_dists) > 0:
                    aggregated_dists[metric].extend(valid_dists)
            except Exception as e:
                print(f"  Error computing {metric} for window {start_idx}: {e}")

    distributions = {}
    
    for metric in metrics:
        valid_dists = np.array(aggregated_dists[metric])
        if len(valid_dists) > 0:
            distributions[metric] = valid_dists
            
            # Calculate key percentiles
            p1 = np.percentile(valid_dists, 1)
            p5 = np.percentile(valid_dists, 5)
            p10 = np.percentile(valid_dists, 10)
            #p25 = np.percentile(valid_dists, 25)
            #p50 = np.percentile(valid_dists, 50)
            
            print(f"  {metric.upper()} Percentiles (across all valid windows):")
            print(f"    1st : {p1:.4f}")
            print(f"    5th : {p5:.4f}")
            print(f"    10th: {p10:.4f}")
            #print(f"    Median: {p50:.4f}")
            
            # Plot Histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(valid_dists, bins=50, kde=True)
            
            # Add percentile lines
            plt.axvline(p1, color='red', linestyle='dashed', linewidth=2, label=f'1st % ({p1:.2f})')
            plt.axvline(p5, color='orange', linestyle='dashed', linewidth=2, label=f'5th % ({p5:.2f})')
            plt.axvline(p10, color='green', linestyle='dashed', linewidth=2, label=f'10th % ({p10:.2f})')
            
            plt.title(f'Distance Distribution: {metric.upper()}\nTarget {product_id} (All Valid Windows)')
            plt.xlabel('Distance')
            plt.ylabel('Frequency (Number of Interactions)')
            plt.legend()
            
            dist_plot_path = os.path.join(plot_dir, f'dist_histogram_all_windows_{metric}.png')
            plt.savefig(dist_plot_path)
            plt.close()
            print(f"  Saved aggregated histogram to {dist_plot_path}")
        else:
            print(f"  Warning: No valid distances returned for {metric} across all windows.")
            
    return distributions

def calculate_thresholds(weights, metric_type, k_std=2.0, p_cutoff=95):
    mean_w = np.mean(weights)
    std_w = np.std(weights)
    median_w = np.median(weights)
    # Approximate MAD
    mad_w = np.median(np.abs(weights - median_w))
    
    thresholds = {}
    if metric_type == "distance":
        # For distances, worst is high. So we cutoff everything ABOVE threshold
        thresholds['Mean + k*Std'] = mean_w + k_std * std_w
        thresholds['Median + k*MAD'] = median_w + (k_std * 1.4826) * mad_w # 1.4826 scales MAD to match std for normal dist
        thresholds[f'Percentile {p_cutoff}'] = np.percentile(weights, p_cutoff)
    else:
        # For similarities, worst is low. So we cutoff everything BELOW threshold
        thresholds['Mean - k*Std'] = mean_w - k_std * std_w
        thresholds['Median - k*MAD'] = median_w - (k_std * 1.4826) * mad_w
        thresholds[f'Percentile {100-p_cutoff}'] = np.percentile(weights, 100 - p_cutoff)
        
    return thresholds

