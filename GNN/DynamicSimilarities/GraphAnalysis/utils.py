import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
import os
import seaborn as sns
import torch
from tslearn.metrics import cdist_dtw
import torch
from tslearn.metrics import dtw
# Graph construction utils

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
                        enable_edges_within_star=True):
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
        raise ValueError("Must provide either a threshold or a percentile.")
        
    if product_id not in df.index:
        raise ValueError(f"Product ID {product_id} not found in DataFrame index.")
        
    time_steps = df.shape[1]
    
    graphs = []
    
    for start_idx in range(0, time_steps - window_size + 1, step_size):
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
                if percentile is not None:
                    k = max(1, int(len(valid_vals) * (percentile / 100.0)))
                    threshold_val = np.partition(valid_vals, k - 1)[k - 1]
                    mask = valid_vals <= threshold_val
                    current_threshold = threshold_val
                else:
                    mask = valid_vals <= threshold
                    current_threshold = threshold
            else: # similarity
                if percentile is not None:
                    k = max(1, int(len(valid_vals) * (percentile / 100.0)))
                    threshold_val = np.partition(valid_vals, -k)[-k]
                    mask = valid_vals >= threshold_val
                    current_threshold = threshold_val
                else:
                    mask = valid_vals >= threshold
                    current_threshold = threshold

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
                                edge_weight = 1.0 / (1.0 + float(val_sub)) 
                                #edge_weight = val_sub
                                G.add_edge(item_ids[idx1], item_ids[idx2], weight=edge_weight)
                        else:
                            if val_sub >= current_threshold:
                                edge_weight = max(0.0, float(val_sub))
                                G.add_edge(item_ids[idx1], item_ids[idx2], weight=edge_weight)

        start_date = str(window_data.columns[0]).split(' ')[0].split('T')[0]
        end_date = str(window_data.columns[-1]).split(' ')[0].split('T')[0]
        
        G.graph['start_date'] = start_date
        G.graph['end_date'] = end_date

        if plot_dir is not None and len(G.nodes) > 1:
            current_plot_dir = plot_dir
            if residuals:
                current_plot_dir = os.path.join(plot_dir, "residuals")
            os.makedirs(current_plot_dir, exist_ok=True)
            
            plot_prefix = "residual_" if residuals else ""
            if not enable_edges_within_star:
                plot_prefix += "star_"
                
            plot_path = os.path.join(current_plot_dir, f'{plot_prefix}graph_{product_id}_{start_date}_to_{end_date}.html')
            try:
                from graph_plot import plot_networkx_plotly
                print(f"Saving plot to {plot_path} with {len(G.nodes)} nodes and {len(G.edges)} edges...")
                plot_networkx_plotly(
                    G, 
                    title=f"Neighbors of Product {product_id} ({metric})<br>Date: {start_date} to {end_date}",
                    save_path=plot_path
                )
            except Exception as e:
                print(f"Plotting skipped due to error: {e}")
                
        graphs.append(G)
                    
    return graphs
def compute_distances_1vsAll(target_ts, all_ts, metric='euclidean', eps=1e-12):
    """
    Computes distances between target_ts (1D) and all_ts (2D) using PyTorch.
    Optimized for 1-vs-all.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.tensor(target_ts, dtype=torch.float32, device=device).unsqueeze(0)
    X = torch.tensor(all_ts, dtype=torch.float32, device=device)
    
    if metric == 'euclidean':
        dist = torch.cdist(target, X, p=2).squeeze(0)
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
        X_np = all_ts if isinstance(all_ts, np.ndarray) else all_ts.cpu().numpy()
        target_np = target_ts if isinstance(target_ts, np.ndarray) else target_ts.cpu().numpy()
        
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
        X_np = all_ts if isinstance(all_ts, np.ndarray) else all_ts.cpu().numpy()
        target_np = target_ts if isinstance(target_ts, np.ndarray) else target_ts.cpu().numpy()
        
        n_X = X_np.shape[0]
        seq_len = X_np.shape[1]
        
        min_dists = np.full(n_X, np.inf)
        
        for shift in range(seq_len):
            shifted_X = np.roll(X_np, shift, axis=1)
            # Compute euclidean distance for this specific shift broadcasted across all rows
            current_dists = np.sqrt(np.sum((shifted_X - target_np)**2, axis=1))
            min_dists = np.minimum(min_dists, current_dists)
            
        return min_dists
        
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
