import sys
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
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
from utils import neighbourhood_graph, compute_similarities_1vsAll, plot_dynamic_graphs
import hashlib



if __name__ == "__main__":
    item_ids = [26008,907969,907967,213626]  # Add your list of product ids here
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
    
    '''
    # --- Apply Per-Product (Row-Wise) Scaling ---
    # Ensures distance metrics (DTW/Euclidean) compare shape rather than raw magnitude
    row_min = df_wide.values.min(axis=1, keepdims=True)
    row_max = df_wide.values.max(axis=1, keepdims=True)
    row_range = np.where(row_max - row_min == 0, 1, row_max - row_min) 
    df_wide.iloc[:, :] = (df_wide.values - row_min) / row_range
    print(f"Applied Per-Product (Row-wise) Min-Max Scaling to df_wide.")
    '''
    window_size = 15 
    step_size = 1
    create_plots = True  # Set to True to enable HTML graph generation
    enable_edges_within_star_opts = [True]  # Grid over excluding vs including edges between neighbors in the star graph
    enable_second_degree_opts = [False]  # Grid over 1st and 2nd degree
    
    grid_configs = [
        #{'metric': 'pearson', 'thresholds': [0.8,0.9, 0.95]},
        #{'metric': 'pearson', 'percentiles':  [0.5, 1, 2]},
        {'metric': 'spearman', 'thresholds': [round(t, 3) for t in np.arange(0.75, 0.85, 0.01)]},
        {'metric': 'pearson', 'thresholds': [round(t, 3) for t in np.arange(0.75, 0.85, 0.01)]},
        {'metric': 'kendall', 'thresholds': [round(t, 3) for t in np.arange(0.75, 0.85, 0.01)]}
    ]
    num_plots_to_draw = None  # Draw all plots if None
    for item_id in item_ids:
        print(f"\n========================================")
        print(f"Processing product ID: {item_id}")
        print(f"========================================")
        
        # Calculate total windows
        total_windows = len(range(0, (train_size + val_size) - window_size + 1, step_size))
        np.random.seed(item_id)  # Seed with item_id for reproducibility across runs
        
        if num_plots_to_draw is None:
            sampled_graph_indices = list(range(total_windows))
        else:
            sampled_graph_indices = np.random.choice(total_windows, size=min(num_plots_to_draw, total_windows), replace=False).tolist()
        
        for config in grid_configs:
            metric = config['metric']
            thresholds = config.get('thresholds', [None])
            percentiles = config.get('percentiles', [None])
            
            for th in thresholds:
                for pct in percentiles:
                    if th is None and pct is None:
                        continue
                        
                    for enable_second_degree in enable_second_degree_opts:
                        for enable_edges_within_star in enable_edges_within_star_opts:
                            prefix = "" if enable_edges_within_star else "star_"
                            if enable_second_degree:
                                prefix = "2nddegree_" + prefix

                            print(f"\n--- Running grid: Metric={metric}, Threshold={th}, Percentile={pct}, 2nd Degree={enable_second_degree}, edges={enable_edges_within_star} ---")
                    
                            # Make safe directory string
                            dir_label = f"pct{pct}" if pct is not None else f"th{th}"
                            
                            # Include the prefix in the folder name so plots don't overwrite each other
                            plot_dir_name = f"{prefix}{dir_label}" if prefix else dir_label
                        
                            if create_plots:
                                plot_output_dir = os.path.join(BASE_DIR, 'GraphPlots', str(item_id), metric, str(window_size), plot_dir_name)
                            else:
                                plot_output_dir = None
                        
                            # Setup PKL path early to check if we can skip
                            pkl_dir = os.path.join(BASE_DIR, "DynamicGraphPkls", str(item_id), metric, str(window_size), str(step_size), plot_dir_name)
                            os.makedirs(pkl_dir, exist_ok=True)
                            pkl_path = os.path.join(pkl_dir, f"{prefix}dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_{dir_label}.pkl")
                                
                            if os.path.exists(pkl_path):
                                if not create_plots:
                                    print(f"Skipping {metric} with {dir_label} - Output PKL already exists!")
                                    continue
                                else:
                                    if os.path.exists(plot_output_dir) and any(f.endswith('.html') for f in os.listdir(plot_output_dir)):
                                        print(f"Skipping {metric} with {dir_label} - Output PKL and plots already exist!")
                                        continue
                                    else:
                                        print(f"PKL exists for {metric} with {dir_label}, but plots are missing. Re-evaluating via neighbourhood_graph...")
                                        
                            graphs, global_threshold = neighbourhood_graph(
                                product_id=item_id,
                                metric_type="similarity",
                                compute_func=compute_similarities_1vsAll, 
                                df=df_wide, 
                                metric=metric, 
                                window_size=window_size, 
                                threshold=th,
                                percentile=pct,
                                step_size=step_size, 
                                cat_labels=cat_labels_dict,
                                plot_dir=plot_output_dir,
                                residuals=False,
                                enable_edges_within_star=enable_edges_within_star,
                                enable_second_degree=enable_second_degree,  # We only want direct neighbors for this analysis
                                num_plots=num_plots_to_draw
                            )
                            
                            valid_graphs = [g for g in graphs if len(g.nodes) > 1]
                            print(f"Finished! Out of {len(graphs)} windows, {len(valid_graphs)} had valid neighbors.")

                            with open(pkl_path, 'wb') as f:
                                pickle.dump(graphs, f)
                            print(f"Successfully saved PKL to {pkl_path}")
                            
                            # Hashing logic purely for generated HTML plots folder
                            if create_plots:
                                try:
                                    is_threshold_mode = th is not None
                                    if is_threshold_mode:
                                        raw_str = "_".join(map(str, thresholds))
                                    else:
                                        raw_str = "_".join(map(str, percentiles))
                                    
                                    html_hash_dir = hashlib.md5(raw_str.encode()).hexdigest()[:8]
                                    
                                    # We preserve original metric inside the PKL logic above,
                                    # but HTML Plotting requires slightly different path formatting
                                    base_plot_dir = os.path.join(BASE_DIR, 'GraphPlots')
                                    # Add the specific threshold/percentile (dir_label) as a subfolder
                                    sub_dir = os.path.join(base_plot_dir, f'window_{window_size}', f'step_{step_size}', f'item_{item_id}', html_hash_dir, dir_label)
                                    os.makedirs(sub_dir, exist_ok=True)
                                    
                                    print(f"Plotting graphs dynamically into HTML files at {sub_dir}...")
                                    
                                    # Extract only the randomly fixed 10 graphs using their specific dates (indices)
                                    graphs_to_plot = [graphs[i] for i in sampled_graph_indices if i < len(graphs)]
                                    
                                    plot_dynamic_graphs(
                                        graphs_to_plot, 
                                        product_id=item_id, 
                                        metric=metric, 
                                        plot_dir=sub_dir, 
                                        residuals=False, 
                                        enable_edges_within_star=enable_edges_within_star, 
                                        enable_second_degree=enable_second_degree, 
                                        num_plots=None, # Already sampled
                                        window_size=window_size, 
                                        step_size=step_size, 
                                        threshold=th if is_threshold_mode else None, 
                                        percentile=pct if not is_threshold_mode else None
                                    )
                                except Exception as plot_e:
                                    print(f"Warning: Plot generation skipped or failed - {plot_e}")
                            