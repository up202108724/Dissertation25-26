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
from utils import analyze_distance_distribution, neighbourhood_graph

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


if __name__ == "__main__":
    item_ids = [26008, 907969,907967]  # Add your list of product ids here
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
    window_size = 7 
    step_size = 1
    grid_configs = [
        {'metric': 'sbd', 'thresholds': [0.02,0.03,0.04]},
    ]
    analyze_distance_distribution_flag = False  # Set to True to analyze distance distributions and get percentile insights before graph construction
    for item_id in item_ids:
        print(f"\n========================================")
        print(f"Processing product ID: {item_id}")
        print(f"========================================")
        
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
        
        for config in grid_configs:
            metric = config['metric']
            for th in config['thresholds']:
                print(f"\n--- Running grid: Metric={metric}, Threshold={th} ---")
                
                # Make safe directory string
                dir_label = f"th{th}"
                plot_output_dir = os.path.join(BASE_DIR, 'GraphPlots','SurveyDistances', str(item_id), metric, dir_label)
                
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
                        metric_type="distance",
                        compute_func=compute_distances_1vsAll, 
                    df=df_wide, 
                    metric=metric, 
                    window_size=window_size, 
                    threshold=th, 
                    step_size=step_size, 
                    cat_labels=cat_labels_dict,
                    plot_dir=plot_output_dir
                )
                    
                valid_graphs = [g for g in graphs if len(g.nodes) > 1]
                print(f"Finished! Out of {len(graphs)} windows, {len(valid_graphs)} had valid neighbors.")
                            
                pkl_dir = os.path.join(BASE_DIR, "DynamicGraphPkls", metric, str(item_id))
                os.makedirs(pkl_dir, exist_ok=True)
                pkl_path = os.path.join(pkl_dir, f"dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_{dir_label}.pkl")
                            
                with open(pkl_path, 'wb') as f:
                    pickle.dump(graphs, f)
                    print(f"Successfully saved PKL to {pkl_path}")