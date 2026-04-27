import sys
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.dirname(__file__)) # Add this directory explicitly

from utils import neighbourhood_graph, compute_distances_1vsAll, calculate_thresholds, plot_dynamic_graphs

if __name__ == "__main__":
    item_ids = [907969]  # Add your list of product ids here
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
    
    distance_metrics = ['cid']
    window_size = 15 
    step_size = 1
    create_plots = False  # Set to True to enable HTML graph generation
    enable_edges_within_star_opts = [True]  # Grid over excluding vs including edges between neighbors in the star graph
    enable_second_degree_opts = [False]  # Grid over 1st and 2nd degree
    num_plots_to_draw = 100 if create_plots else None  # Specify number of random plots here
    grid_configs = [
        # Example for thresholds:
        #{'metric': 'hamming', 'thresholds': [0.143, 0.286]},
        # Example for percentiles (top 0.5%, 1%, 2% closest items):
        #{'metric': 'euclidean', 'percentiles': [0.5, 1, 2]},
        #{'metric': 'hamming', 'percentiles': [0.5, 1, 2]},
        #{'metric': 'amplitude_offset', 'percentiles': [0.5, 1, 2]},
        #{'metric': 'slope_consistency', 'percentiles': [0.5, 1, 2]},
        {'metric': 'cid', 'percentiles': [0.5, 1, 2]},
        #{'metric': 'dtw', 'percentiles': [0.5, 1, 2]},
        #{'metric': 'phase_invariance', 'percentiles': [0.5, 1, 2]}
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
                        
                            csv_path_check = pkl_path.replace('.pkl', '_worst_distances.csv')
                            if os.path.exists(pkl_path) and os.path.exists(csv_path_check):
                                if not create_plots:
                                    print(f"Skipping {metric} with {dir_label} - Output PKL and CSV already exist!")
                                    continue
                                else:
                                    if os.path.exists(plot_output_dir) and any(f.endswith('.html') for f in os.listdir(plot_output_dir)):
                                        print(f"Skipping {metric} with {dir_label} - Output PKL and plots already exist!")
                                        continue
                                    else:
                                        print(f"PKL exists for {metric} with {dir_label}, but plots are missing. Re-evaluating via neighbourhood_graph...")
                                        
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
                                plot_dir=plot_output_dir,
                                residuals=False,
                                enable_edges_within_star=enable_edges_within_star,
                                enable_second_degree=enable_second_degree,  # We only want direct neighbors for this analysis
                                num_plots=num_plots_to_draw
                            )
                            
                            valid_graphs = [g for g in graphs if len(g.nodes) > 1]
                            print(f"Finished! Out of {len(graphs)} windows, {len(valid_graphs)} had valid neighbors.")
                        
                            # -- Analysis: Distribution of furthest distances --
                            worst_edges = []
                            for i, g in enumerate(graphs):
                                if len(g.edges) > 0:
                                    max_dist = max([data['weight'] for u, v, data in g.edges(data=True)])
                                else:
                                    max_dist = np.nan
                                worst_edges.append(max_dist)
                                
                            df_worst = pd.DataFrame({'window_idx': range(len(graphs)), 'worst_distance': worst_edges})
                            csv_path = pkl_path.replace('.pkl', '_worst_distances.csv')
                            df_worst.to_csv(csv_path, index=False)
                            print(f"Saved worst distances distribution to {csv_path}")
                            
                            plt.figure(figsize=(10, 5))
                            plt.hist(df_worst['worst_distance'].dropna(), bins=30, alpha=0.7, color='red')
                            plt.title(f'Distribution of Maximum Graph Distances\nMetric: {metric}, Pct: {pct}')
                            plt.xlabel('Maximum Distance in Graph')
                            plt.ylabel('Frequency')
                            plt.grid(True, alpha=0.3)
                            plot_path = pkl_path.replace('.pkl', '_worst_distances_hist.png')
                            plt.savefig(plot_path)
                            plt.close()

                            with open(pkl_path, 'wb') as f:
                                pickle.dump(graphs, f)
                            print(f"Successfully saved PKL to {pkl_path}")
                            
                            # --- 5. Adaptive Threshold Filtering ---
                            print("\n--- Applying Adaptive Thresholds ---")
                            import re
                            
                            all_weights = []
                            for g in graphs:
                                all_weights.extend([data.get('weight', 0) for u, v, data in g.edges(data=True)])
                            
                            # Determine metric_type based on whether the original logic targets "distance"
                            curr_metric_type = "distance" 
                            
                            if len(all_weights) > 0:
                                candidate_thresholds = calculate_thresholds(all_weights, curr_metric_type, k_std=1.5, p_cutoff=90)
                                
                                for strat_name, thr_val in candidate_thresholds.items():
                                    safe_strat_name = re.sub(r'[^a-zA-Z0-9_\-]', '', strat_name.replace(' ', '_'))
                                    
                                    strat_pkl_dir = os.path.join(pkl_dir, safe_strat_name)
                                    os.makedirs(strat_pkl_dir, exist_ok=True)
                                    strat_pkl_path = os.path.join(strat_pkl_dir, os.path.basename(pkl_path))
                                    
                                    filtered_graphs = []
                                    total_edges_before = 0
                                    total_edges_after = 0
                                    
                                    for g in graphs:
                                        g_filtered = g.copy()
                                        total_edges_before += g.number_of_edges()
                                        
                                        if curr_metric_type == "distance":
                                            edges_to_remove = [(u, v) for u, v, data in g.edges(data=True) if data.get('weight', 0) > thr_val]
                                        else:
                                            edges_to_remove = [(u, v) for u, v, data in g.edges(data=True) if data.get('weight', 0) < thr_val]
                                            
                                        g_filtered.remove_edges_from(edges_to_remove)
                                        filtered_graphs.append(g_filtered)
                                        total_edges_after += g_filtered.number_of_edges()
                                        
                                    print(f"Strategy: {strat_name} (Thresh: {thr_val:.4f}). Edges retained: {total_edges_after}/{total_edges_before}")
                                    
                                    # Save Filtered Graphs PKL
                                    with open(strat_pkl_path, 'wb') as f:
                                        pickle.dump(filtered_graphs, f)
                                        
                                    # Calculate Worst Distances for Filtered Graphs
                                    worst_edges_filt = []
                                    for g in filtered_graphs:
                                        if len(g.edges) > 0:
                                            max_dist = max([data['weight'] for u, v, data in g.edges(data=True)])
                                        else:
                                            max_dist = np.nan
                                        worst_edges_filt.append(max_dist)
                                        
                                    df_worst_filt = pd.DataFrame({'window_idx': range(len(filtered_graphs)), 'worst_distance': worst_edges_filt})
                                    csv_path_filt = strat_pkl_path.replace('.pkl', '_worst_distances.csv')
                                    df_worst_filt.to_csv(csv_path_filt, index=False)
                                    
                                    # HTML Plots logic if create_plots is on
                                    if create_plots:
                                        strat_plot_dir = os.path.join(plot_output_dir, safe_strat_name) if plot_output_dir else None
                                        if strat_plot_dir:
                                            os.makedirs(strat_plot_dir, exist_ok=True)
                                            plot_dynamic_graphs(
                                                graphs=filtered_graphs,
                                                product_id=item_id,
                                                metric=metric,
                                                plot_dir=strat_plot_dir,
                                                residuals=False,
                                                enable_edges_within_star=enable_edges_within_star,
                                                enable_second_degree=enable_second_degree,
                                                num_plots=num_plots_to_draw,
                                                window_size=window_size,
                                                step_size=step_size,
                                                threshold=thr_val,
                                                percentile=pct
                                            )
                            else:
                                print("No edges found across all graphs, skipping adaptive thresholding.")