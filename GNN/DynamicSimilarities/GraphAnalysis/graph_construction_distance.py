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

from utils import neighbourhood_graph, compute_distances_1vsAll

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
    
    distance_metrics = ['euclidean', 'hamming', 'amplitude_offset', 'slope_consistency', 'cid', 'dtw', 'phase_invariance']
    window_size = 15 
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
    
    for item_id in item_ids:
        print(f"\n========================================")
        print(f"Processing product ID: {item_id}")
        print(f"========================================")
        
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
                    plot_output_dir = os.path.join(BASE_DIR, 'GraphPlots', str(item_id), metric, str(window_size), dir_label)
                    
                    # Setup PKL path early to check if we can skip
                    pkl_dir = os.path.join(BASE_DIR, "DynamicGraphPkls", metric, str(item_id))
                    os.makedirs(pkl_dir, exist_ok=True)
                    pkl_path = os.path.join(pkl_dir, f"dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_{dir_label}.pkl")
                    
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
                        metric_type="distance",
                        compute_func=compute_distances_1vsAll, 
                        df=df_wide, 
                        metric=metric, 
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