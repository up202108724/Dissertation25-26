import sys
import os
import pickle
import time
import torch
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from invariances import compute_temporal_distances

def compute_dynamic_distances(
    df: pd.DataFrame,
    date_col: str,
    item_col: str,
    target_col: str,
    window_size: int,
    step_size: int,
    distance_method: str = "dtw",  # "euclidean", "dtw", "cid", "hamming"
    aggfunc: str = "sum",
    **kwargs
):
    """
    Computes a sequence of dynamic distance matrices using a sliding window 
    over the dates in the dataframe.
    """
    unique_dates = sorted(df[date_col].unique())
    num_dates = len(unique_dates)
    
    dist_dfs = []
    df_pivots = []
    window_info = []

    for start_idx in range(0, num_dates - window_size + 1, step_size):
        end_idx = start_idx + window_size
        current_dates = unique_dates[start_idx:end_idx]
        
        start_date = current_dates[0]
        end_date = current_dates[-1]
        
        mask = df[date_col].isin(current_dates)
        df_window = df[mask]
        
        print(f"\nComputing {distance_method} distances for window: {start_date} to {end_date}")
        
        # Maps "standardscaled" to "euclidean" for backwards compatibility
        mapped_method = "euclidean" if distance_method == "standardscaled" else distance_method

        dist_df, df_scaled = compute_temporal_distances(
            df=df_window,
            date_col=date_col,
            item_col=item_col,
            target_col=target_col,
            distance_method=mapped_method,
            aggfunc=aggfunc,
            **kwargs
        )
        
        dist_dfs.append(dist_df)
        df_pivots.append(df_scaled)  # Storing the scaled/pivot differences
        window_info.append({"start_date": start_date, "end_date": end_date})
        
    return dist_dfs, df_pivots, window_info

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'data_andre.feather')
    
    DATE_COL = 'date'
    TARGET_COL = 'value'
    ITEM_COL = 'item_id'
    
    WINDOW_SIZE = 7
    STEP_SIZE = 1
    DISTANCE_METHODS = ["euclidean", "cid", "hamming", "amplitude_offset", "slope_consistency"] # Choose distance method
    NUM_ITEMS = None # Limit for faster testing 
    

    PERCENTILE_LIST = [0.1, 0.25, 0.5, 0.75] # Percentiles to compute for distance distribution analysis
    
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

    for DISTANCE_METHOD in DISTANCE_METHODS:
        print(f"\n================ Processing method: {DISTANCE_METHOD} ================\n")
        output_dir = os.path.join(os.path.dirname(__file__), f"DynamicGraphPkls/{DISTANCE_METHOD}")
        os.makedirs(output_dir, exist_ok=True)
        
        OUTPUT_PATH = os.path.join(output_dir, f'dynamic_invariances_output_{DISTANCE_METHOD}_Window{WINDOW_SIZE}_Step{STEP_SIZE}.pkl')
        
        DISTANCES_PATH = os.path.join(output_dir, f'dynamic_distances_{DISTANCE_METHOD}_Window{WINDOW_SIZE}_Step{STEP_SIZE}.pkl')
        
        if os.path.exists(DISTANCES_PATH):
            print(f"File {DISTANCES_PATH} already exists. Loading data from pickle...")
            with open(DISTANCES_PATH, 'rb') as f:
                distance_data = pickle.load(f)
            dist_dfs = distance_data['dist_dfs']
            df_pivots = distance_data['df_pivots']
            window_info = distance_data['window_info']
        else:
            print(f"Computing dynamic {DISTANCE_METHOD} distance matrices...")
            start_time = time.time()
            
            dist_dfs, df_pivots, window_info = compute_dynamic_distances(
                df_train_val,
                date_col=DATE_COL,
                item_col=ITEM_COL,
                target_col=TARGET_COL,
                window_size=WINDOW_SIZE,
                step_size=STEP_SIZE,
                distance_method=DISTANCE_METHOD
            )
            
            end_time_distances = time.time()
            print(f"Time taken to compute distances: {end_time_distances - start_time:.2f} seconds")
            
            distance_data = {
                'dist_dfs': dist_dfs,
                'df_pivots': df_pivots,
                'window_info': window_info
            }
            
            print(f"Exporting distances to {DISTANCES_PATH}...")
            with open(DISTANCES_PATH, 'wb') as f:
                pickle.dump(distance_data, f)
            
        # Plot boxplot for distance values
        # Flatten the upper triangular part (excluding diagonal) of each distance matrix
        all_dist_values = []
        for dist_df in dist_dfs:
            if isinstance(dist_df, pd.DataFrame):
                arr = dist_df.values
            else:
                arr = dist_df
                
            # get upper triangle without diagonal
            idx = np.triu_indices_from(arr, k=1)
            vals = arr[idx]
            
            # Filter out NaNs if any and cast to float32 to save memory
            vals = vals[~pd.isna(vals)].astype(np.float32)
            
            # Subsample values to prevent ArrayMemoryError (keep up to 10k random samples per window)
            if len(vals) > 10000:
                vals = np.random.choice(vals, size=10000, replace=False)
                
            all_dist_values.extend(vals)
        
        if len(all_dist_values) > 0:
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=all_dist_values)
            plt.title(f'Distance Boxplot\nMethod: {DISTANCE_METHOD} | Window: {WINDOW_SIZE}')
            plt.ylabel('Distance Value')
            
            for percentile in PERCENTILE_LIST:
                p_value = np.percentile(all_dist_values, percentile)
                plt.axhline(p_value, color='r', linestyle='--', label=f'{int(percentile)}th p: {p_value:.2f}')
            plt.legend()
            
            boxplot_path = os.path.join(output_dir, f'{WINDOW_SIZE}_{STEP_SIZE}_{DISTANCE_METHOD}_boxplot.png')
            plt.savefig(boxplot_path)
            plt.close()
            print(f"Saved boxplot to {boxplot_path}")
        else:
            print("No distance values to plot.")
            
    print("Done! Data exported successfully for all methods.")
