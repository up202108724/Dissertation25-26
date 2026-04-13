import sys
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.dirname(__file__)) # Add this directory explicitly
from model_utils.utils import generate_exogenous_features
try:
    from graph_plot import plot_networkx_plotly
except ImportError:
    try:
        from GNN.DynamicSimilarities.GraphAnalysis.graph_plot import plot_networkx_plotly
    except ImportError:
        pass # Handle it if neither works without breaking inference unless they actually call plot functions

import torch
from tslearn.metrics import dtw
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, HuberRegressor
from utils import neighbourhood_graph, compute_similarities_1vsAll

def compute_residuals_per_sku(
    df_long,
    train_end_date,
    target_col='value',
    item_col='item_id',
    date_col='date',
    model_type='ridge',
    alpha=1.0,
    standardize=True
):
    """
    Computes regression residuals for each SKU to factor out calendar/seasonal/promo impacts.
    """
    df = df_long.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    train_mask_global = df[date_col] <= pd.to_datetime(train_end_date)

    EXOG_COLS = [
        # base
        "day_of_week", "day_of_month", "week_of_year", "week_of_month",
        "month", "quarter", "is_weekend",
        "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",

        # special days of the week
        "is_monday", "is_friday",
        # holidays
        "is_holiday", "is_thanksgiving", "is_black_friday",
        "is_christmas", "is_christmas_eve", "is_new_year_eve",
        "is_pre_holiday_1", "is_pre_holiday_2", "is_pre_holiday_3", "is_pre_holiday_7",
        "is_post_holiday_1", "is_post_holiday_2", "is_post_holiday_3", "is_post_holiday_7",

        # boundary / behavior
        "is_bridge_day",

        # promotions
        "promo_type_FRPG", "promo_value_FRPG",
        "promo_type_GAS", "promo_value_GAS",
        "promo_type_BOGO", "promo_value_BOGO",
        "promo_type_DISC", "promo_value_DISC",
        "promo_type_CIRC", "promo_value_CIRC",
        "promo_type_CIRE", "promo_value_CIRE",
        "promo_type_CLCP", "promo_value_CLCP",
        "promo_type_LFPE", "promo_value_LFPE",
        
        #trend indicators are handled separately to avoid data leakage
        "time_idx"
    ]
    
    # 1. Generate the exogenous features utilizing the updated builder function
    df = generate_exogenous_features(df, exog_cols=EXOG_COLS, date_col=date_col)

    df['y_hat'] = np.nan
    df['residual'] = np.nan
    
    feature_cols = EXOG_COLS

    # 2. Iterate per SKU to fit a local Ridge regression
    for sku, gidx in df.groupby(item_col).groups.items():
        idx = list(gidx)
        # sub represents the time sequence for the current SKU
        sub = df.loc[idx].sort_values(date_col)

        train_mask = sub[date_col] <= pd.to_datetime(train_end_date)
        
        # Fallback if there are not enough rows to train a regression
        if train_mask.sum() < 10:
            mean_val = sub.loc[train_mask, target_col].mean()
            if pd.isna(mean_val):
                mean_val = 0.0
            
            df.loc[sub.index, 'y_hat'] = mean_val
            df.loc[sub.index, 'residual'] = sub[target_col] - mean_val
            continue

        # Target X and Y for regression
        X_train = sub.loc[train_mask, feature_cols].fillna(0.0).values
        y_train = sub.loc[train_mask, target_col].astype(float).values

        # Predict across ALL historical data for the SKU (train + val + test)
        X_all = sub[feature_cols].fillna(0.0).values
        y_all = sub[target_col].astype(float).values

        # Select regression model
        if model_type == 'lasso':
            model = Lasso(alpha=alpha, max_iter=2000)
        elif model_type == 'elasticnet':
            model = ElasticNet(alpha=alpha, max_iter=2000)
        elif model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'huber':
            model = HuberRegressor(max_iter=2000)
        else: # default ridge
            model = Ridge(alpha=alpha)
            
        model.fit(X_train, y_train)

        # Compute predictions and deduct from real values to get the isolated residual 
        y_hat = model.predict(X_all)
        resid = y_all - y_hat

        df.loc[sub.index, 'y_hat'] = y_hat
        df.loc[sub.index, 'residual'] = resid

    # 3. Standardize Residuals (z-scoring based on train set properties)
    if standardize:
        train_stats = (
            df.loc[train_mask_global]
              .groupby(item_col)['residual']
              .agg(train_resid_mean='mean', train_resid_std='std')
              .reset_index()
        )
        
        # Merge stats dynamically
        df = df.merge(train_stats, on=item_col, how='left')
        
        # Avoid division-by-zero (fallback to simple residual shift if variance is essentially zero)
        df['residual_std'] = np.where(
            df['train_resid_std'] > 1e-8,
            (df['residual'] - df['train_resid_mean']) / df['train_resid_std'],
            df['residual'] - df['train_resid_mean']
        )
        value_for_wide = 'residual_std'
    else:
        value_for_wide = 'residual'

    # 4. Pivot sequence data into wide format
    residuals_wide = (
        df.pivot(index=item_col, columns=date_col, values=value_for_wide)
          .sort_index()
          .sort_index(axis=1)
    )
    return residuals_wide, df



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
    
    val_end_date = df_wide.columns[-1]  # Get the last date from the restricted wide dataframe
    
    # Detrend data to get residuals
    print(f"Computing residuals (detrending raw signal up to {val_end_date}) ...")
    
    # compute_residuals_per_sku requires the long-format dataframe, so we filter it to match the same dates
    df_long_filtered = df[df['date'] <= val_end_date].copy()
    model_type_ = 'ridge'  # You can experiment with 'lasso', 'elasticnet', 'linear', 'huber'
    df_residuals_wide, df_residuals_long = compute_residuals_per_sku(
        df_long=df_long_filtered, 
        train_end_date=val_end_date, 
        model_type=model_type_, 
        alpha=1.0, 
        standardize=True
    )
    # Fill any NaNs created by rolling mean
    df_residuals = df_residuals_wide.fillna(0)
    
    similarity_metrics = ['pearson', 'spearman', 'kendall']
    window_sizes = [21, 28, 35, 42, 56]
    step_size = 1
    create_plots = True  # Set to True to enable HTML graph generation
    
    grid_configs = [
        {'metric': 'pearson', 'top_k_percents': [0.5, 1, 2 ]}, # e.g. top 5%
        {'metric': 'spearman', 'top_k_percents': [0.5, 1 ,2]}
    ]
    
    for item_id in item_ids:
        print(f"\n========================================")
        print(f"Processing product ID: {item_id}")
        print(f"========================================")
        
        for window_size in window_sizes:
            for config in grid_configs:
                metric = config['metric']
                for top_k in config['top_k_percents']:
                    print(f"\n--- Running grid: Window={window_size}, Metric={metric}, Top-K={top_k}% ---")
                    
                    # Make safe directory string
                    dir_label = f"top{top_k}pct"
                    
                    if create_plots:
                        plot_output_dir = os.path.join(BASE_DIR, 'GraphPlots',f"residuals_{model_type_}", str(item_id), str(window_size), str(step_size), metric, dir_label)
                        
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
                        df=df_residuals,  # Crucial step: Use the residual dataframe!
                        metric=metric,
                        metric_type="similarity",
                        window_size=window_size,
                        compute_func=compute_similarities_1vsAll,
                        percentile=top_k, 
                        step_size=step_size, 
                        cat_labels=cat_labels_dict,
                        plot_dir=plot_output_dir,
                        residuals=True
                    )
                    
                    valid_graphs = [g for g in graphs if len(g.nodes) > 1]
                    print(f"Finished {metric} top-{top_k}% (Win: {window_size}) for item {item_id}! Out of {len(graphs)} windows, {len(valid_graphs)} had valid neighbors.")
                                
                    pkl_dir = os.path.join(BASE_DIR, "DynamicGraphPkls", f"residuals_{model_type_}", metric, str(item_id), str(window_size), str(step_size))
                    os.makedirs(pkl_dir, exist_ok=True)
                    pkl_path = os.path.join(pkl_dir, f"dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_{dir_label}.pkl")
                                
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(graphs, f)
                        print(f"Successfully saved PKL to {pkl_path}")