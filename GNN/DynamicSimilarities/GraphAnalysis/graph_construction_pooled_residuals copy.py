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
from utils import neighbourhood_graph

def compute_similarities_1vsAll(target_ts, all_ts, metric='pearson', eps=1e-12):
    """
    Computes similarities between target_ts (1D) and all_ts (2D) using PyTorch.
    Optimized for 1-vs-all.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.tensor(target_ts, dtype=torch.float32, device=device).unsqueeze(0)
    X = torch.tensor(all_ts, dtype=torch.float32, device=device)
    
    if metric == 'pearson':
        target_mean = torch.mean(target, dim=1, keepdim=True)
        X_mean = torch.mean(X, dim=1, keepdim=True)
        
        target_centered = target - target_mean
        X_centered = X - X_mean
        
        cov = torch.sum(target_centered * X_centered, dim=1)
        target_var = torch.sqrt(torch.sum(target_centered**2, dim=1))
        X_var = torch.sqrt(torch.sum(X_centered**2, dim=1))
        
        sim = cov / (target_var * X_var + eps)
        return sim.cpu().numpy()
        
    elif metric == 'spearman':
        _, target_indices = torch.sort(target, dim=1)
        target_ranks = torch.empty_like(target)
        target_ranks.scatter_(1, target_indices, torch.arange(1, target.shape[1]+1, dtype=torch.float32, device=device).unsqueeze(0))
        
        _, X_indices = torch.sort(X, dim=1)
        X_ranks = torch.empty_like(X)
        X_ranks.scatter_(1, X_indices, torch.arange(1, X.shape[1]+1, dtype=torch.float32, device=device).unsqueeze(0).expand_as(X))
        
        target_mean = torch.mean(target_ranks, dim=1, keepdim=True)
        X_mean = torch.mean(X_ranks, dim=1, keepdim=True)
        
        target_centered = target_ranks - target_mean
        X_centered = X_ranks - X_mean
        
        cov = torch.sum(target_centered * X_centered, dim=1)
        target_var = torch.sqrt(torch.sum(target_centered**2, dim=1))
        X_var = torch.sqrt(torch.sum(X_centered**2, dim=1))
        
        sim = cov / (target_var * X_var + eps)
        return sim.cpu().numpy()
        
    elif metric == 'kendall':
        seq_len = target.shape[1]
        if seq_len < 2:
            return torch.ones(X.shape[0], device=device).cpu().numpy()
            
        idx1, idx2 = torch.triu_indices(seq_len, seq_len, offset=1, device=device)
        
        target_diffs = target[:, idx1] - target[:, idx2]
        target_signs = torch.sign(target_diffs)
        
        X_diffs = X[:, idx1] - X[:, idx2]
        X_signs = torch.sign(X_diffs)
        
        S = torch.sum(target_signs * X_signs, dim=1)
        
        target_non_ties = torch.sum(target_signs**2, dim=1)
        X_non_ties = torch.sum(X_signs**2, dim=1)
        
        denom = torch.sqrt(target_non_ties * X_non_ties)
        
        sim = torch.where(denom == 0, torch.tensor(0.0, device=device), S / denom)
        return sim.cpu().numpy()
        
    else:
        raise ValueError(f"Metric {metric} not supported")

def compute_residuals(df_wide):
    """
    Computes residuals for the time series dataframe.
    Subtracts a 7-day moving average and day-of-week average (simple baseline).
    """
    # 7-day rolling mean to remove local trend
    trend = df_wide.rolling(window=7, axis=1, min_periods=1, center=True).mean()
    detrended = df_wide - trend
    
    # Optional: further remove day of week seasonality if dates are known, but for simplicity
    # just return the detrended series.
    return detrended

def compute_residuals_pooled(
    df_long,
    train_end_date,
    target_col='value',
    item_col='item_id',
    date_col='date',
    promo_cols=None,
    extra_numeric_cols=None,
    use_log1p=False,
    alpha=3.0,
    standardize=True
):
    """
    Ajusta uma regressão pooled entre SKUs com intercepto específico por SKU
    (via one-hot de item_id) + efeitos comuns de calendário/promo/trend.

    Retorna:
        residuals_wide: DataFrame wide [rows=item_id, cols=date]
        fitted_pipe: pipeline sklearn ajustado
        df_out: DataFrame long com y_hat, residual, residual_std
    """
    df = df_long.copy()
    df = add_calendar_features(df, date_col=date_col)

    if promo_cols is None:
        promo_cols = [c for c in df.columns if c.startswith('promo_')]

    if extra_numeric_cols is None:
        extra_numeric_cols = []

    categorical_cols = [item_col, 'dow', 'month']
    numeric_cols = ['trend', 'is_month_start', 'is_month_end', 'is_weekend'] + promo_cols + extra_numeric_cols

    # garantir tipos
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # target
    y_raw = df[target_col].astype(float).values
    if use_log1p:
        y = np.log1p(y_raw)
    else:
        y = y_raw

    # split treino
    df[date_col] = pd.to_datetime(df[date_col])
    train_mask = df[date_col] <= pd.to_datetime(train_end_date)

    X = df[categorical_cols + numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0.0))
            ]), numeric_cols)
        ]
    )

    model = Ridge(alpha=alpha)

    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])

    pipe.fit(X.loc[train_mask], y[train_mask])

    y_hat = pipe.predict(X)

    if use_log1p:
        # resíduos na escala original
        y_hat_orig = np.expm1(y_hat)
        residual = y_raw - y_hat_orig
        df['y_hat'] = y_hat_orig
    else:
        residual = y_raw - y_hat
        df['y_hat'] = y_hat

    df['residual'] = residual

    # padronização por SKU com estatísticas APENAS do treino
    if standardize:
        train_stats = (
            df.loc[train_mask]
              .groupby(item_col)['residual']
              .agg(train_resid_mean='mean', train_resid_std='std')
        )

        df = df.join(train_stats, on=item_col)

        # evitar divisões por zero
        df['train_resid_std'] = df['train_resid_std'].fillna(0.0)
        df['residual_std'] = np.where(
            df['train_resid_std'] > 1e-8,
            (df['residual'] - df['train_resid_mean']) / df['train_resid_std'],
            0.0
        )
        value_for_wide = 'residual_std'
    else:
        value_for_wide = 'residual'

    residuals_wide = (
        df.pivot(index=item_col, columns=date_col, values=value_for_wide)
          .sort_index()
          .sort_index(axis=1)
    )

    return residuals_wide, pipe, df


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
    
    # Detrend data to get residuals
    print("Computing residuals (detrending raw signal) ...")
    df_residuals = compute_residuals(df_wide)
    # Fill any NaNs created by rolling mean
    df_residuals = df_residuals.fillna(0)
    
    similarity_metrics = ['pearson', 'spearman', 'kendall']
    window_sizes = [21, 28, 35, 42, 56]
    step_size = 1
    create_plots = False  # Set to True to enable HTML graph generation
    
    grid_configs = [
        {'metric': 'pearson', 'top_k_percents': [1, 2.5, 5, 10]}, # e.g. top 5%
        {'metric': 'spearman', 'top_k_percents': [1, 5, 10]}
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
                        # Crucial step: Use the residual dataframe!
                        df=df_residuals, 
                        metric=metric, 
                        window_size=window_size, 
                        top_k_percent=top_k, 
                        step_size=step_size, 
                        cat_labels=cat_labels_dict,
                        plot_dir=plot_output_dir
                    )
                    
                    valid_graphs = [g for g in graphs if len(g.nodes) > 1]
                    print(f"Finished {metric} top-{top_k}% (Win: {window_size}) for item {item_id}! Out of {len(graphs)} windows, {len(valid_graphs)} had valid neighbors.")
                                
                    pkl_dir = os.path.join(BASE_DIR, "DynamicGraphPkls", metric, str(item_id), str(window_size), str(step_size))
                os.makedirs(pkl_dir, exist_ok=True)
                pkl_path = os.path.join(pkl_dir, f"dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_{dir_label}.pkl")
                            
                with open(pkl_path, 'wb') as f:
                    pickle.dump(graphs, f)
                    print(f"Successfully saved PKL to {pkl_path}")