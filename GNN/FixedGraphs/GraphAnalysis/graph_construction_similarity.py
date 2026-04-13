import sys
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.dirname(__file__)) # Add this directory explicitly
try:
    from graph_plot import plot_networkx_plotly
except ImportError:
    try:
        from GNN.DynamicSimilarities.GraphAnalysis.graph_plot import plot_networkx_plotly
    except ImportError:
        pass # Handle it if neither works without breaking inference unless they actually call plot functions

import torch
from tslearn.metrics import dtw

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

def neighbourhood_graph(product_id, df, similarity_metric, window_size, threshold, step_size=1, cat_labels=None, plot_dir=None):
    """
    Constructs a graph by iterating over sliding time windows of the time series data.
    Finds items within the specified similarity metric thresholds to product_id.
    
    Args:
        product_id: ID of the central product
        df: DataFrame with time series (rows=items, cols=time steps)
        similarity_metric: Similarity metric string (e.g. 'pearson')
        window_size: Size of the sliding window
        threshold: Minimum similarity to be considered a neighbor

        step_size: Number of timesteps the window slides forward
        cat_labels: Dictionary or Series mapping item IDs to category labels (optional)
        plot_dir: If provided, saves the constructed graphs as HTML files in this directory
    """
    if product_id not in df.index:
        raise ValueError(f"Product ID {product_id} not found in DataFrame index.")
        
    time_steps = df.shape[1]
    print(f"Constructing graph for product {product_id} with window size {window_size}, step size {step_size}, threshold {threshold} ({time_steps} time steps)...")
    
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
        
        # Compute 1-vs-all similarities for the current window
        similarities = compute_similarities_1vsAll(target_ts, all_ts, metric=similarity_metric)
        
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
        valid_similarities = similarities[valid_mask]
        valid_original_idxs = np.arange(len(item_ids))[valid_mask]
        
        mask = valid_similarities >= threshold
        selected_similarities = valid_similarities[mask]
        selected_ids = valid_item_ids[mask]
        selected_orig_idxs = valid_original_idxs[mask]
        current_threshold = threshold

        neighbor_indices = []
        for orig_idx, sim_val, other_id in zip(selected_orig_idxs, selected_similarities, selected_ids):
            neighbor_indices.append((orig_idx, other_id))
            if not G.has_node(other_id):
                cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
                G.add_node(other_id, cat_label=cat_other)
            G.add_edge(product_id, other_id, weight=float(sim_val))
                
        # Connect first-degree neighbors to each other if they respect the current_threshold
        if len(neighbor_indices) > 1:
            n_ts_array = np.array([all_ts[idx] for idx, _ in neighbor_indices])
            n_ids = [oid for _, oid in neighbor_indices]
            
            for i, n_id1 in enumerate(n_ids):
                n1_ts = n_ts_array[i]
                n_sims = compute_similarities_1vsAll(n1_ts, n_ts_array, metric=similarity_metric)
                for j in range(i + 1, len(n_ids)):
                    if n_sims[j] >= current_threshold:
                        # Extra safeguard: Don't connect 0 flatlines to each other
                        if np.sum(np.abs(n1_ts)) > 0 and np.sum(np.abs(n_ts_array[j])) > 0:
                            G.add_edge(n_id1, n_ids[j], weight=float(n_sims[j]))

        start_date = str(window_data.columns[0]).split('T')[0]
        end_date = str(window_data.columns[-1]).split('T')[0]
        
        # Attach for later reference as well
        G.graph['start_date'] = start_date
        G.graph['end_date'] = end_date

        if G.number_of_edges() == 0:
            print(f"No edges created for window {start_date} to {end_date}.")
            
        if plot_dir is not None and len(G.nodes) > 1:
            os.makedirs(plot_dir, exist_ok=True)
            print(f"Constructed graph for window {start_date} to {end_date} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            safe_start_date = start_date.replace(':', '-').replace(' ', '_')
            save_path = os.path.join(plot_dir, f"graph_{safe_start_date}.html")
            
            plot_networkx_plotly(
                G, 
                title=f"Graph from {start_date} to {end_date} (Method: {similarity_metric}, Threshold: {threshold})", 
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
    
    similarity_metrics = ['pearson', 'spearman', 'kendall']
    window_size = df_wide.shape[1] # Fixed graph, single window over entire timeframe
    step_size = 1
    create_plots = True  # Set to True to enable HTML graph generation
    
    grid_configs = [
        {'metric': 'pearson', 'thresholds': [0.8,0.9, 0.95]},
        {'metric': 'spearman', 'thresholds': [0.7,0.8, 0.9]},
        {'metric': 'kendall', 'thresholds': [0.7, 0.80]}
    ]
    
    for config in grid_configs:
        metric = config['metric']
        for th in config['thresholds']:
            print(f"\n--- Running grid: Metric={metric}, Threshold={th} ---")
            
            # Make safe directory string
            dir_label = f"th{th}"
            
            if create_plots:
                plot_output_dir = os.path.join(BASE_DIR, 'FixedGraphPlots', str(item_id), metric, dir_label)
                
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
            else:
                plot_output_dir = None
                    
            graphs = neighbourhood_graph(
                product_id=item_id, 
                df=df_wide, 
                similarity_metric=metric, 
                window_size=window_size, 
                threshold=th, 
                step_size=step_size, 
                cat_labels=cat_labels_dict,
                plot_dir=plot_output_dir
            )
            
            valid_graphs = [g for g in graphs if len(g.nodes) > 1]
            print(f"Finished {metric} th={th}! Out of {len(graphs)} windows, {len(valid_graphs)} had valid neighbors.")
                        
            pkl_dir = os.path.join(BASE_DIR, "FixedGraphPkls", metric, str(item_id))
            os.makedirs(pkl_dir, exist_ok=True)
            pkl_path = os.path.join(pkl_dir, f"fixed_graph_{metric}_{dir_label}.pkl")
                        
            with open(pkl_path, 'wb') as f:
                pickle.dump(graphs, f)
                print(f"Successfully saved PKL to {pkl_path}")