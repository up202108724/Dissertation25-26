import pickle
import os
import sys
import pandas as pd
import numpy as np
import time 

from dynamic_invariances import build_dynamic_distance_graphs
from dynamic_similarities import build_dynamic_similarity_graphs
from graph_analysis import generate_and_save_graph_plots, generate_and_save_node_stats

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'data_andre.feather')
    
    DATE_COL = 'date'
    TARGET_COL = 'value'
    ITEM_COL = 'item_id'
    
    WINDOW_SIZE = 7
    STEP_SIZE = 1
    NUM_ITEMS = None # Limit for faster testing 
    
    # ------------------------------------------------------------------------------------------
    # GRID SEARCH CONFIGURATION
    # Define which methods to run and their specific thresholds
    # ------------------------------------------------------------------------------------------
    grid_config = {
        "similarities": {
            "kendall": [0.7, 0.8, 0.9],
            "spearman": [0.9, 0.95, 0.99],
            "pearson": [0.8, 0.85, 0.9],
            # Add more similarity methods and their specific thresholds here
        },
        "distances": {
            "cid": [0.5, 0.75, 1.0],
            "amplitude_offset": [0.5, 1.0, 1.5],
            "slope_consistency": [0.01, 0.05, 0.1],
            "euclidean": [5.0, 10.0, 15.0]
            # Add more invariant distance methods and their specific thresholds here
        }
    }
    
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_feather(DATA_PATH)
    
    if DATE_COL not in df.columns:
        if DATE_COL in df.index.names:
            df = df.reset_index(level=DATE_COL)
        else:
            df = df.reset_index()
            if DATE_COL not in df.columns and DATE_COL.capitalize() in df.columns:
                df = df.rename(columns={DATE_COL.capitalize(): DATE_COL})

    if NUM_ITEMS is not None:
        top_items = df[ITEM_COL].unique()[:NUM_ITEMS]
        df = df[df[ITEM_COL].isin(top_items)]
        print(f"Using top {NUM_ITEMS} products.")
    
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, ITEM_COL]).reset_index(drop=True)
    
    TEST_SIZE = 152
    unique_dates = df[DATE_COL].dropna().unique()
    train_val_dates = unique_dates[:-TEST_SIZE]
    df_train_val = df[df[DATE_COL].isin(train_val_dates)].copy()
    
    # Assemble all scenarios based on dynamic config
    scenarios = []
    
    for method, thresh_list in grid_config.get("similarities", {}).items():
        for th in thresh_list:
            scenarios.append({
                'type': 'similarity',
                'method': method,
                'threshold': th,
                'window_size': WINDOW_SIZE,
                'step_size': STEP_SIZE
            })
            
    for method, thresh_list in grid_config.get("distances", {}).items():
        for th in thresh_list:
            scenarios.append({
                'type': 'distance',
                'method': method,
                'threshold': th,
                'window_size': WINDOW_SIZE,
                'step_size': STEP_SIZE
            })

    print(f"Starting grid analysis over {len(scenarios)} configurations...\n")
    
    for config in scenarios:
        method_type = config['type']
        method_name = config['method']
        threshold = config['threshold']
        w = config['window_size']
        s = config['step_size']
        
        print(f"\n{'='*60}")
        print(f"Executing: {method_name.upper()} ({method_type}) - Threshold: {threshold}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Define output directory for PKLs inside DynamicGraphPkls
        output_dir = os.path.join(os.path.dirname(__file__), 'DynamicGraphPkls', method_name)
        os.makedirs(output_dir, exist_ok=True)
        
        if method_type == 'distance':
            metric_type = "Distance"
            base_name = f'dynamic_invariances_output_{method_name}_{threshold}_Window{w}_Step{s}'
            pkl_path = os.path.join(output_dir, f'{base_name}.pkl')
            plot_dir = os.path.join(output_dir, f'{base_name}_plots')
            stats_csv_path = os.path.join(output_dir, f'node_stats_{base_name}.csv')
            
            print(f"Building dynamic '{method_name}' distance graphs...")
            graphs, dist_dfs, df_pivots, window_info = build_dynamic_distance_graphs(
                df_train_val,
                date_col=DATE_COL,
                item_col=ITEM_COL,
                target_col=TARGET_COL,
                window_size=w,
                step_size=s,
                distance_method=method_name,
                distance_threshold=threshold
            )
            
            output_data = {
                'graphs': graphs,
                'dist_dfs': dist_dfs,
                'df_pivots': df_pivots,
                'window_info': window_info,
                'distance_method': method_name,
                'distance_threshold': threshold,
            }
            
        else: # similarity
            metric_type = "Similarity"
            base_name = f'dynamic_graphs_output_{method_name}_{threshold}_Window{w}_Step{s}'
            pkl_path = os.path.join(output_dir, f'{base_name}.pkl')
            plot_dir = os.path.join(output_dir, f'{base_name}_plots')
            stats_csv_path = os.path.join(output_dir, f'node_stats_{base_name}.csv')
            
            print(f"Building dynamic '{method_name}' similarity graphs...")
            graphs, sim_dfs, df_pivots, window_info, feature_dfs = build_dynamic_similarity_graphs(
                df_train_val,
                date_col=DATE_COL,
                item_col=ITEM_COL,
                target_col=TARGET_COL,
                window_size=w,
                step_size=s,
                similarity_method=method_name,
                similarity_threshold=threshold
            )
            
            output_data = {
                'graphs': graphs,
                'sim_dfs': sim_dfs,
                'dynamic_graph_features': feature_dfs,
                'df_pivots': df_pivots,
                'window_info': window_info,
                'similarity_method': method_name,
                'similarity_threshold': threshold,
            }

        end_time = time.time()
        print(f"Graph construction taken: {end_time - start_time:.2f} seconds")

        print(f"Exporting results to {pkl_path}...")
        with open(pkl_path, 'wb') as f:
            pickle.dump(output_data, f)
            
        print(f"Generating Analysis Plots and Stats for {method_name}...")
        sim_or_dist_dfs = output_data.get('sim_dfs', output_data.get('dist_dfs'))
        
        # 1. Generate plots
        generate_and_save_graph_plots(
            graphs=graphs,
            sim_dfs=sim_or_dist_dfs,
            plot_dir=plot_dir,
            method_name=method_name,
            metric_type=metric_type
        )
        
        # 2. Generate stats
        generate_and_save_node_stats(
            graphs=graphs,
            output_csv_path=stats_csv_path
        )
        
    print("\n🏁 Unified grid search analysis and graph generation completed!")