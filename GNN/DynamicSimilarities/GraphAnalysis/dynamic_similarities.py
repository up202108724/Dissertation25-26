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

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from calculate_node_features import compute_window_node_features
from similarities import compute_temporal_similarities

def compute_dynamic_similarities(
    df: pd.DataFrame,
    date_col: str,
    item_col: str,
    target_col: str,
    window_size: int,
    step_size: int,
    aggfunc: str = "sum",
    similarity_method: str = "pearson"
):
    """
    Computes a sequence of dynamic similarity matrices using a sliding window 
    over the dates in the dataframe.
    """
    unique_dates = sorted(df[date_col].unique())
    num_dates = len(unique_dates)
    
    sim_dfs = []
    df_pivots = []
    window_info = []

    for start_idx in range(0, num_dates - window_size + 1, step_size):
        end_idx = start_idx + window_size
        current_dates = unique_dates[start_idx:end_idx]
        
        start_date = current_dates[0]
        end_date = current_dates[-1]
        
        mask = df[date_col].isin(current_dates)
        df_window = df[mask]
        
        print(f"\nComputing similarities for window: {start_date} to {end_date}")
        
        sim_df, df_pivot = compute_temporal_similarities(
            df=df_window,
            date_col=date_col,
            item_col=item_col,
            target_col=target_col,
            aggfunc=aggfunc,
            similarity_method=similarity_method
        )
        
        sim_dfs.append(sim_df)
        df_pivots.append(df_pivot)
        window_info.append({"start_date": start_date, "end_date": end_date})
        
    return sim_dfs, df_pivots, window_info

if __name__ == "__main__":
    # 1. Definir caminhos e hiperparâmetros
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'data_andre.feather')
    
    DATE_COL = 'date'
    TARGET_COL = 'value'
    ITEM_COL = 'item_id'
    
    WINDOW_SIZES = [7]
    STEP_SIZE = 1
    SIMILARITY_METHODS = ["pearson", "spearman", "kendall", "theil-sen"]
    NUM_ITEMS = None
    COMPUTE_NODE_FEATURES = False
    USE_ABSOLUTE_SIMILARITY = False  # Added so negative correlations (substitute products) are considered equally central
    
    print(f"Loading data from {DATA_PATH}...")
    df_raw = pd.read_feather(DATA_PATH)
    
    # Handle DATE_COL
    if DATE_COL not in df_raw.columns:
        if DATE_COL in df_raw.index.names:
            df_raw = df_raw.reset_index(level=DATE_COL)
        else:
            df_raw = df_raw.reset_index()
            if DATE_COL not in df_raw.columns and DATE_COL.capitalize() in df_raw.columns:
                df_raw = df_raw.rename(columns={DATE_COL.capitalize(): DATE_COL})
            elif DATE_COL not in df_raw.columns and DATE_COL.upper() in df_raw.columns:
                df_raw = df_raw.rename(columns={DATE_COL.upper(): DATE_COL})

    if NUM_ITEMS is not None:
        top_items = df_raw[ITEM_COL].unique()[:NUM_ITEMS]
        df_raw = df_raw[df_raw[ITEM_COL].isin(top_items)]
        print(f"Using top {NUM_ITEMS} products.")
    else:
        print("NUM_ITEMS is None. Using all available products.")
    
    df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL])
    df_raw = df_raw.sort_values([DATE_COL, ITEM_COL]).reset_index(drop=True)
    
    TEST_SIZE = 152
    unique_dates = df_raw[DATE_COL].dropna().unique()
    train_val_dates = unique_dates[:-TEST_SIZE]
    df_train_val = df_raw[df_raw[DATE_COL].isin(train_val_dates)].copy()
    
    

    for SIMILARITY_METHOD in SIMILARITY_METHODS:
        for WINDOW_SIZE in WINDOW_SIZES:
                print(f"\n==========================================")
                print(f"Running grid for: Method={SIMILARITY_METHOD}, Window={WINDOW_SIZE}")
                print(f"==========================================")
                
                output_dir = os.path.join(os.path.dirname(__file__), "DynamicGraphPkls", SIMILARITY_METHOD)
                os.makedirs(output_dir, exist_ok=True)
                
                SIMILARITIES_PATH = os.path.join(output_dir, f'dynamic_similarities_{SIMILARITY_METHOD}_Window{WINDOW_SIZE}_Step{STEP_SIZE}.pkl')
                OUTPUT_PATH = os.path.join(output_dir, f'dynamic_graphs_output_{SIMILARITY_METHOD}_Window{WINDOW_SIZE}_Step{STEP_SIZE}.pkl')
                
                if os.path.exists(SIMILARITIES_PATH):
                    print(f"File {SIMILARITIES_PATH} already exists. Loading data from pickle...")
                    with open(SIMILARITIES_PATH, 'rb') as f:
                        sim_data = pickle.load(f)
                    sim_dfs = sim_data['sim_dfs']
                    df_pivots = sim_data['df_pivots']
                    window_info = sim_data['window_info']
                else:
                    start_time = time.time()
                    sim_dfs, df_pivots, window_info = compute_dynamic_similarities(
                        df_train_val,
                        date_col=DATE_COL,
                        item_col=ITEM_COL,
                        target_col=TARGET_COL,
                        window_size=WINDOW_SIZE,
                        step_size=STEP_SIZE,
                        similarity_method=SIMILARITY_METHOD
                    )
                    end_time_sims = time.time()
                    print(f"Time taken to compute similarities: {end_time_sims - start_time:.2f} seconds")
                    
                    sim_data = {
                        'sim_dfs': sim_dfs,
                        'df_pivots': df_pivots,
                        'window_info': window_info,
                    }
                    print(f"Exporting similarities to {SIMILARITIES_PATH}...")
                    with open(SIMILARITIES_PATH, 'wb') as f:
                        pickle.dump(sim_data, f)
                
                # Plot boxplot for similarity values
                # Flatten the upper triangular part (excluding diagonal) of each similarity matrix
                all_sim_values = []
                for sim_df in sim_dfs:
                    # sim_df is an Item x Item numpy array or pandas DataFrame
                    if isinstance(sim_df, pd.DataFrame):
                        arr = sim_df.values
                    else:
                        arr = sim_df
                        
                    # get upper triangle without diagonal
                    idx = np.triu_indices_from(arr, k=1)
                    vals = arr[idx]
                    
                    # Filter out NaNs if any and cast to float32 to save memory
                    vals = vals[~pd.isna(vals)].astype(np.float32)
                    
                    if USE_ABSOLUTE_SIMILARITY:
                        vals = np.abs(vals)
                    
                    # Subsample values to prevent ArrayMemoryError (keep up to 10k random samples per window)
                    if len(vals) > 10000:
                        vals = np.random.choice(vals, size=10000, replace=False)
                        
                    all_sim_values.extend(vals)
                
                if len(all_sim_values) > 0:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(y=all_sim_values)
                    plt.title(f'Similarity Boxplot\nMethod: {SIMILARITY_METHOD} | Window: {WINDOW_SIZE}')
                    plt.ylabel('Similarity Value')
                    
                    # Compute percentiles and draw horizontal lines
                    p99 = np.percentile(all_sim_values, 99)
                    p99_5 = np.percentile(all_sim_values, 99.5)
                    p99_9 = np.percentile(all_sim_values, 99.9)
                    
                    plt.axhline(p99, color='r', linestyle='--', label=f'99th p: {p99:.2f}')
                    plt.axhline(p99_5, color='g', linestyle='--', label=f'99.5th p: {p99_5:.2f}')
                    plt.axhline(p99_9, color='b', linestyle='--', label=f'99.9th p: {p99_9:.2f}')
                    plt.legend(loc='upper right')
                    
                    boxplot_path = os.path.join(output_dir, f'{WINDOW_SIZE}_{STEP_SIZE}_{USE_ABSOLUTE_SIMILARITY}_boxplot.png')
                    plt.savefig(boxplot_path)
                    plt.close()
                    print(f"Saved boxplot to {boxplot_path}")
                
                else:
                    print("No similarity values to plot.")

    print("Done! All configurations processed.")