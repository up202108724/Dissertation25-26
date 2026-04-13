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
from utils import neighbourhood_graph, compute_similarities_1vsAll




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
    
    similarity_metrics = ['pearson', 'spearman', 'kendall']
    window_size = 15 
    step_size = 1
    create_plots = True  # Set to True to enable HTML graph generation
    
    grid_configs = [
        {'metric': 'pearson', 'thresholds': [0.8,0.9, 0.95]},
        {'metric': 'spearman', 'thresholds': [0.7,0.8, 0.9]},
        {'metric': 'kendall', 'thresholds': [0.7, 0.80]}
    ]
    
    for item_id in item_ids:
        print(f"\n========================================")
        print(f"Processing product ID: {item_id}")
        print(f"========================================")
        
        for config in grid_configs:
            metric = config['metric']
            for th in config['thresholds']:
                print(f"\n--- Running grid: Metric={metric}, Threshold={th} ---")
                
                # Make safe directory string
                dir_label = f"th{th}"
                
                if create_plots:
                    plot_output_dir = os.path.join(BASE_DIR, 'GraphPlots', str(item_id), str(window_size), str(step_size), metric, dir_label)
                    
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
                        metric_type="similarity",
                        compute_func=compute_similarities_1vsAll, 
                    df=df_wide, 
                    metric=metric, 
                    window_size=window_size, 
                    threshold=th, 
                    step_size=step_size, 
                    cat_labels=cat_labels_dict,
                    plot_dir=plot_output_dir
                )
                
                valid_graphs = [g for g in graphs if len(g.nodes) > 1]
                print(f"Finished {metric} th={th} for item {item_id}! Out of {len(graphs)} windows, {len(valid_graphs)} had valid neighbors.")
                            
                pkl_dir = os.path.join(BASE_DIR, "DynamicGraphPkls", metric, str(item_id), str(window_size), str(step_size))
                os.makedirs(pkl_dir, exist_ok=True)
                pkl_path = os.path.join(pkl_dir, f"dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_{dir_label}.pkl")
                            
                with open(pkl_path, 'wb') as f:
                    pickle.dump(graphs, f)
                    print(f"Successfully saved PKL to {pkl_path}")