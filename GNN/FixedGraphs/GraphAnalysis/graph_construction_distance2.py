import sys
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from graph_plot import plot_networkx_plotly
import torch
from tslearn.metrics import dtw
from utils import analyze_distance_distribution
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

def compute_distances_1vsAll(target_ts, all_ts, metric='euclidean', eps=1e-12):
    """
    Computes distances between target_ts (1D) and all_ts (2D) using PyTorch.
    Optimized for 1-vs-all.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.tensor(target_ts, dtype=torch.float32, device=device).unsqueeze(0)
    X = torch.tensor(all_ts, dtype=torch.float32, device=device)

    if metric == 'lorentzian':
        # Lorentzian distance: sum(ln(1 + |x - y|))
        # Robust against outliers and noise
        diffs = torch.abs(X - target) + 1
        dist = torch.sum(torch.log1p(diffs), dim=1)
        return dist.cpu().numpy()
        
    elif metric == 'sbd':
        # Shape-Based Distance (SBD) using cross-correlation via FFT
        # Normalizes inputs to mean 0, std 1 usually, but here we do raw or local normalized depending on need.
        # SBD = 1 - max(NCC)
        target_np = target_ts if isinstance(target_ts, np.ndarray) else target_ts.cpu().numpy()
        X_np = all_ts if isinstance(all_ts, np.ndarray) else all_ts.cpu().numpy()
        
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
        # Fallback to tslearn/sktime if possible, otherwise basic DP is needed.
        # As MSM requires complex DP and specific cost parameters (c), 
        # we try to import sktime's msm_distance, else fallback to something accessible.
        try:
            from sktime.distances import msm_distance
            X_np = all_ts if isinstance(all_ts, np.ndarray) else all_ts.cpu().numpy()
            target_np = target_ts if isinstance(target_ts, np.ndarray) else target_ts.cpu().numpy()
            return np.array([msm_distance(target_np, x) for x in X_np])
        except ImportError:
            raise ValueError("metric='msm' requires 'sktime' to be installed. Run `pip install sktime`.")
            
    elif metric == 'edr' or metric == 'lcss':
        # Edit Distance on Real sequence (EDR) / Longest Common Subsequence (LCSS)
        # Often implemented via tslearn or sktime.
        # Sktime has both EDR and LCSS which operate with an epsilon for matching.
        target_np = target_ts if isinstance(target_ts, np.ndarray) else target_ts.cpu().numpy()
        X_np = all_ts if isinstance(all_ts, np.ndarray) else all_ts.cpu().numpy()
        
        if metric == 'lcss':
            from tslearn.metrics import lcss
            # Note: LCSS returns similarity ([0, 1]), so distance is 1 - similarity
            # Requires eps parameter; tslearn uses esp=1.0 by default or scaled.
            epsilon = 0.5 # A common threshold relative to scale; can be tuned
            return np.array([1 - lcss(target_np, x, eps=epsilon) for x in X_np])
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

def neighbourhood_graph(product_id, df, distance_metric, window_size, threshold, step_size=1, cat_labels=None, plot_dir=None):
    """
    Constructs a graph by iterating over sliding time windows of the time series data.
    Finds items within the specified distance metric thresholds to product_id.
    
    Args:
        product_id: ID of the central product
        df: DataFrame with time series (rows=items, cols=time steps)
        distance_metric: Distance metric string (e.g. 'euclidean')
        window_size: Size of the sliding window
        threshold: Maximum distance to be considered a neighbor
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
        
        mask = valid_distances <= threshold
        selected_distances = valid_distances[mask]
        selected_ids = valid_item_ids[mask]
        selected_orig_idxs = valid_original_idxs[mask]
        current_threshold = threshold

        neighbor_indices = []
        for orig_idx, dist, other_id in zip(selected_orig_idxs, selected_distances, selected_ids):
            neighbor_indices.append((orig_idx, other_id))
            if not G.has_node(other_id):
                cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
                G.add_node(other_id, cat_label=cat_other)
            G.add_edge(product_id, other_id, weight=float(dist))
                
        # Connect first-degree neighbors to each other if they respect the current_threshold
        if len(neighbor_indices) > 1:
            n_ts_array = np.array([all_ts[idx] for idx, _ in neighbor_indices])
            n_ids = [oid for _, oid in neighbor_indices]
            
            for i, n_id1 in enumerate(n_ids):
                n1_ts = n_ts_array[i]
                n_dists = compute_distances_1vsAll(n1_ts, n_ts_array, metric=distance_metric)
                for j in range(i + 1, len(n_ids)):
                    if n_dists[j] <= current_threshold:
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
            
            plot_networkx_plotly(
                G, 
                 title=f"Graph from {start_date} to {end_date} (Method: {distance_metric}, Distance: {threshold})",
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
    
    distance_metrics = ['lorentzian', 'sbd', 'msm']
    window_size = df_wide.shape[1] # Fixed graph, single window over entire timeframe
    step_size = 1
    analyze_distance_distribution_flag = True  # Set to True to enable distance distribution analysis and plotting
    # Analyze Distance Distributions
    hist_dir = os.path.join(BASE_DIR, 'GraphPlots', 'SurveyDistances', str(item_id), 'Histograms')
    if analyze_distance_distribution_flag == True:
        analyze_distance_distribution(
            df=df_wide, 
            product_id=item_id, 
            window_size=window_size, 
            metrics=distance_metrics, 
            plot_dir=hist_dir
        )
    
    grid_configs = [
        {'metric': 'sbd', 'thresholds': [0.02,0.03,0.04]},
    ]
    
    for config in grid_configs:
        metric = config['metric']
        for th in config['thresholds']:
            print(f"\n--- Running grid: Metric={metric}, Threshold={th} ---")
            
            # Make safe directory string
            dir_label = f"th{th}"
            plot_output_dir = os.path.join(BASE_DIR, 'FixedGraphPlots','SurveyDistances', str(item_id), metric, dir_label)
            
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
                step_size=step_size, 
                cat_labels=cat_labels_dict,
                plot_dir=plot_output_dir
            )
                
    valid_graphs = [g for g in graphs if len(g.nodes) > 1]
    print(f"Finished! Out of {len(graphs)} windows, {len(valid_graphs)} had valid neighbors.")
                
    pkl_dir = os.path.join(BASE_DIR, "FixedGraphPkls", metric, str(item_id))
    os.makedirs(pkl_dir, exist_ok=True)
    pkl_path = os.path.join(pkl_dir, f"fixed_graph_{metric}_{dir_label}.pkl")
                
    with open(pkl_path, 'wb') as f:
        pickle.dump(graphs, f)
        print(f"Successfully saved PKL to {pkl_path}")