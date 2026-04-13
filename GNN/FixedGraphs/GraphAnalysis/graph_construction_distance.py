import sys
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from graph_plot import plot_networkx_plotly
from tslearn.metrics import cdist_dtw
import torch
from tslearn.metrics import dtw

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

def neighbourhood_graph(product_id, df, distance_metric, window_size, threshold=None, percentile=None, step_size=1, cat_labels=None, plot_dir=None):
    """
    Constructs a graph by iterating over sliding time windows of the time series data.
    Finds items within the specified distance metric thresholds or percentile to product_id.
    
    Args:
        product_id: ID of the central product
        df: DataFrame with time series (rows=items, cols=time steps)
        distance_metric: Distance metric string (e.g. 'euclidean')
        window_size: Size of the sliding window
        threshold: Maximum distance to be considered a neighbor
        percentile: Top % of closest items to be considered neighbors (e.g., 5 for top 5%)
        step_size: Number of timesteps the window slides forward
        cat_labels: Dictionary or Series mapping item IDs to category labels (optional)
        plot_dir: If provided, saves the constructed graphs as HTML files in this directory
    """
    if threshold is None and percentile is None:
        raise ValueError("Must provide either a threshold or a percentile.")
        
    if product_id not in df.index:
        raise ValueError(f"Product ID {product_id} not found in DataFrame index.")
        
    time_steps = df.shape[1]
    print(f"Constructing graph for product {product_id} with window size {window_size}, step size {step_size}, threshold {threshold}, percentile {percentile} ({time_steps} time steps)...")
    
    graphs = []
    
    for start_idx in range(0, time_steps - window_size + 1, step_size):
        G = nx.Graph()
        
        # Add target node with its category if available
        cat = cat_labels.get(product_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
        G.add_node(product_id, cat_label=cat)
        
        end_idx = start_idx + window_size
        window_data = df.iloc[:, start_idx:end_idx]
        
        # Extract target and all other time series
        target_ts = window_data.loc[product_id].values
        all_ts = window_data.values
        item_ids = window_data.index.values
        
        # Compute 1-vs-all distances for the current window
        distances = compute_distances_1vsAll(target_ts, all_ts, metric=distance_metric)
        
        # Apply filtering rules
        # FIX: Ignore products that had absolutely zero sales in this specific window!
        # Measuring similarity against a flat "dead" line creates massive meaningless zero-distance clusters (especially in Hamming).
        active_items_mask = np.sum(np.abs(all_ts), axis=1) > 0
        valid_mask = (item_ids != product_id) & active_items_mask
        
        # If the target product itself has absolutely 0 sales in this 7-day window, 
        # it mathematically shouldn't have "shape" neighbors. Force mask to False.
        if np.sum(np.abs(target_ts)) == 0:
            valid_mask = np.zeros_like(valid_mask, dtype=bool)
            
        valid_item_ids = item_ids[valid_mask]
        valid_distances = distances[valid_mask]
        valid_original_idxs = np.arange(len(item_ids))[valid_mask]
        
        if len(valid_distances) == 0:
            continue
            
        if percentile is not None:
            # We want the 'least distance', so we look at the lower percentile (e.g., 5th percentile)
            current_threshold = np.percentile(valid_distances, percentile)
        else:
            current_threshold = threshold
        
        mask = valid_distances <= current_threshold
        selected_distances = valid_distances[mask]
        selected_ids = valid_item_ids[mask]
        selected_orig_idxs = valid_original_idxs[mask]

        neighbor_indices = []
        for orig_idx, dist, other_id in zip(selected_orig_idxs, selected_distances, selected_ids):
            neighbor_indices.append((orig_idx, other_id))
            if not G.has_node(other_id):
                cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
                G.add_node(other_id, cat_label=cat_other)
            G.add_edge(product_id, other_id, weight=float(dist))
                
        # Connect first-degree neighbors to each other if they respect the current_threshold
        if len(neighbor_indices) > 1:
            # The worst distance among the connected neighbors to the ego node
            worst_distance_threshold = np.max(selected_distances)
            
            n_ts_array = np.array([all_ts[idx] for idx, _ in neighbor_indices])
            n_ids = [oid for _, oid in neighbor_indices]
            
            for i, n_id1 in enumerate(n_ids):
                n1_ts = n_ts_array[i]
                n_dists = compute_distances_1vsAll(n1_ts, n_ts_array, metric=distance_metric)
                for j in range(i + 1, len(n_ids)):
                    # Use the worst distance to the ego node as the threshold to connect neighbors to each other
                    if n_dists[j] <= worst_distance_threshold:
                        # Extra safeguard: Don't connect 0 flatlines to each other
                        if np.sum(np.abs(n1_ts)) > 0 and np.sum(np.abs(n_ts_array[j])) > 0:
                            G.add_edge(n_id1, n_ids[j], weight=float(n_dists[j]))

        if plot_dir is not None and len(G.nodes) > 1:
            os.makedirs(plot_dir, exist_ok=True)
            start_date = str(window_data.columns[0]).split('T')[0]
            end_date = str(window_data.columns[-1]).split('T')[0]
            
            # Attach for later reference as well
            G.graph['start_date'] = start_date
            G.graph['end_date'] = end_date
            print(f"Constructed graph for window {start_date} to {end_date} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            safe_start_date = start_date.replace(':', '-').replace(' ', '_')
            save_path = os.path.join(plot_dir, f"graph_{safe_start_date}.html")
            
            method_desc = f"Percentile: {percentile}%" if percentile is not None else f"Distance: {threshold}"
            
            plot_networkx_plotly(
                G, 
                 title=f"Graph from {start_date} to {end_date} (Method: {distance_metric}, {method_desc})",
                save_path=save_path,
                target_node=product_id
            )
                
        graphs.append(G)
                    
    return graphs

if __name__ == "__main__":
    item_id = 907969
    # Use absolute path to ensure it finds the dataset regardless of where the script is executed from
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', '..', '..', 'dataset', 'data_andre.feather')
    print(f"Loading data from {DATA_PATH}...")
    # Use pd.read_feather instead of feather.read_table to get a DataFrame directly
    df = pd.read_feather(DATA_PATH)
    
    # --- 1. Extract category dictionary automatically ---
    # We can just extract it directly from the dataset without external files!
    cat_labels_dict = df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()

    # --- 2. Pivot the data to wide format (rows=items, cols=time steps) ---
    df_wide = df.pivot_table(index='item_id', columns='date', values='value', aggfunc='sum').fillna(0)
    
    # Use only train and val sets for building graphs
    train_size = 455
    val_size = 154
    df_wide = df_wide.iloc[:, :train_size + val_size]
    
    distance_metrics = ['euclidean', 'hamming', 'amplitude_offset', 'slope_consistency', 'cid', 'dtw', 'phase_invariance']
    window_size = df_wide.shape[1] # Fixed graph, single window over entire timeframe
    step_size = 1
    
    grid_configs = [
        # Example for thresholds:
        #{'metric': 'hamming', 'thresholds': [0.143, 0.286]},
        # Example for percentiles (top 0.5%, 1%, 2% closest items):
        {'metric': 'euclidean', 'percentiles': [0.5, 1, 2]},
        #{'metric': 'hamming', 'percentiles': [0.5, 1, 2]},
        {'metric': 'amplitude_offset', 'percentiles': [0.5, 1, 2]},
        {'metric': 'slope_consistency', 'percentiles': [0.5, 1, 2]},
        {'metric': 'cid', 'percentiles': [0.5, 1, 2]},
        {'metric': 'dtw', 'percentiles': [0.5, 1, 2]},
        {'metric': 'phase_invariance', 'percentiles': [0.5, 1, 2]}
    ]
    
    for config in grid_configs:
        metric = config['metric']
        thresholds = config.get('thresholds', [None])
        percentiles = config.get('percentiles', [None])
        
        for th in thresholds:
            for pct in percentiles:
                if th is None and pct is None:
                    continue
                    
                print(f"\n--- Running grid: Metric={metric}, Threshold={th}, Percentile={pct} ---")
                
                # Make safe directory string
                dir_label = f"pct{pct}" if pct is not None else f"th{th}"
                plot_output_dir = os.path.join(BASE_DIR, 'FixedGraphPlots', str(item_id), metric, dir_label)
                
                # Setup PKL path early to check if we can skip
                pkl_dir = os.path.join(BASE_DIR, "FixedGraphPkls", metric, str(item_id))
                os.makedirs(pkl_dir, exist_ok=True)
                pkl_path = os.path.join(pkl_dir, f"fixed_graph_{metric}_{dir_label}.pkl")
                
                if os.path.exists(pkl_path) and os.path.exists(plot_output_dir):
                    print(f"Skipping {metric} with {dir_label} - Output PKL and plot directory already exist!")
                    continue
                
                if os.path.exists(plot_output_dir):
                    import shutil
                    import time
                    import stat
                    
                    def force_remove(func, path, exc_info):
                        try:
                            os.chmod(path, stat.S_IWRITE)
                            func(path)
                        except Exception:
                            pass
                            
                    for _ in range(5):
                        try:
                            shutil.rmtree(plot_output_dir, onerror=force_remove)
                            if not os.path.exists(plot_output_dir):
                                break
                        except Exception:
                            time.sleep(0.5)
                    
                    if os.path.exists(plot_output_dir):
                        os.system(f'rmdir /S /Q "{plot_output_dir}"')
                        
                graphs = neighbourhood_graph(
                    product_id=item_id, 
                    df=df_wide, 
                    distance_metric=metric, 
                    window_size=window_size, 
                    threshold=th,
                    percentile=pct,
                    step_size=step_size, 
                    cat_labels=cat_labels_dict,
                    plot_dir=plot_output_dir
                )
                    
                valid_graphs = [g for g in graphs if len(g.nodes) > 1]
                print(f"Finished! Out of {len(graphs)} windows, {len(valid_graphs)} had valid neighbors.")
                
                with open(pkl_path, 'wb') as f:
                    pickle.dump(graphs, f)
                print(f"Successfully saved PKL to {pkl_path}")