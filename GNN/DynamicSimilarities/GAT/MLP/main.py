import sys
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import networkx as nx
from utils import compute_distances_1vsAll, compute_similarities_1vsAll
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add LSTM directory to path to reuse training scripts, datasets and plots

from plots import plot_results
from utils import generate_exogenous_features
from train import TrainConfig, train_mlp_forecaster
from graphsageinference import recursive_inference, recursive_inference_no_graph
from utils import compute_distances_1vsAll, compute_similarities_1vsAll, neighbourhood_graph
import itertools
import csv
import hashlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def infer_metric_type(metric):
    distance_metrics = ['euclidean','manhattan', 'hamming', 'amplitude_offset', 'slope_consistency', 'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
    if metric in distance_metrics:
        return 'distance'
    else:
        return 'similarity'

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join(script_dir, '..', '..', '..', '..', 'dataset', 'data_andre.feather')
DATE_COL = 'date'
TARGET_COL = 'value'

train_size = 579
val_size = 30
forecast_horizon = 152
lookback_window = 30

EXOG_COLS = [
    "dow_sin","dow_cos","doy_sin","doy_cos","is_weekend",
    "rolling_mean_7",
    "month", "quarter",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_monday", "is_friday",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_bridge_day",
]

grid_configs = [
    # Distâncias Robustas e Lock-step
    #{'metric': 'cid', 'percentiles': [0.5, 1, 2]},
    #{'metric': 'amplitude_offset', 'percentiles': [0.5, 1, 2]},
    {'metric': 'spearman', 'thresholds': [round(t, 3) for t in np.arange(0.82, 0.92, 0.03)]},
    #{'metric': 'cid', 'thresholds': [round(t, 2) for t in np.arange(2.0, 3.2, 0.01)]},
    # Distâncias Robustas e Lock-step
    #{'metric': 'amplitude_offset', 'thresholds': [round(t, 2) for t in np.arange(2.0, 3.5, 0.01)]},
]


batch_size = 32
hidden_sizes = (64, 32)
dropout = 0.2
EPOCHS = 1000
LEARNING_RATE = 0.001
SEEDS = [42]
window_sizes = [15]     
step_sizes = [1]
enable_edges_opts = [True]
enable_second_degree_opts = [False]  # We will keep this False for the main analysis, but you can set to True to include second-degree neighbors in the graph construction
USE_RESIDUALS = False
MODEL_TYPE = 'ridge'
loss_type = 'MSELoss'
PATIENCE = 150
SAVE_MODELS = False
SAVE_PLOTS = True
USE_EMBEDDINGS = True
SAVE_EMBEDDINGS = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PRODUCTS_TO_TEST = [
  (26008, 6269),
  (907969, 6269),
  (907967, 6269),
  (213626, 6269)
]

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_feather(DATA_PATH)
    
    if DATE_COL in df.index.names:
        df = df.reset_index(drop=True) if DATE_COL in df.columns else df.reset_index()
    if df.index.name == DATE_COL:
         df = df.reset_index(drop=True)
         
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, "item_id", "store_id"]).reset_index(drop=True)
    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS)
    full_df = df.copy()

    # Pre-generate df_wide and category labels for inference
    cat_labels_dict = full_df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict() if 'cat_label' in full_df.columns else {}
    df_wide_global = full_df.pivot_table(index='item_id', columns=DATE_COL, values=TARGET_COL, aggfunc='sum').fillna(0)
    df_wide_global.columns = pd.to_datetime(df_wide_global.columns).strftime('%Y-%m-%d')

    # Globally define train split based on identical logic for all items
    L = len(df_wide_global.columns)
    global_train_start_idx = L - forecast_horizon - val_size - train_size
    global_val_start_idx = L - forecast_horizon - val_size

    # Build dictionary of local StandardScalers and transform df_wide_scaled for distances
    if global_train_start_idx < 0: global_train_start_idx = 0
    train_df_wide = df_wide_global.iloc[:, global_train_start_idx:global_val_start_idx]
    
    df_wide_scaled = df_wide_global.copy()
    product_scalers = {}
    for item_id_iter in df_wide_global.index:
        z_scaler = StandardScaler()
        train_ts = train_df_wide.loc[item_id_iter].values.reshape(-1, 1)
        z_scaler.fit(train_ts)
        product_scalers[item_id_iter] = z_scaler
        full_ts = df_wide_global.loc[item_id_iter].values.reshape(-1, 1)
        df_wide_scaled.loc[item_id_iter] = z_scaler.transform(full_ts).flatten()

    os.makedirs('best_models', exist_ok=True)
    os.makedirs('grid_search_plots', exist_ok=True)
    
    for product_id, store_id in PRODUCTS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"PROCESSING PRODUCT {product_id} FOR STORE {store_id}")
        print(f"{'='*80}\n")
        
        df_product = full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)].copy()
        df_product[DATE_COL] = pd.to_datetime(df_product[DATE_COL])
        df_product = df_product.sort_values(DATE_COL).reset_index(drop=True)
        
        test_start_idx = len(df_product) - forecast_horizon
        val_start_idx = test_start_idx - val_size
        train_start_idx = val_start_idx - train_size

        train_slice = slice(train_start_idx, val_start_idx)
        val_slice = slice(val_start_idx, test_start_idx)
        test_slice = slice(test_start_idx, None)
        
        train = df_product[TARGET_COL][train_slice].values
        val = df_product[TARGET_COL][val_slice].values
        test = df_product[TARGET_COL][test_slice].values
        
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
        val_scaled = scaler.transform(val.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test.reshape(-1, 1)).flatten()

        if len(EXOG_COLS) > 0:
            exog_train = df_product[EXOG_COLS][train_slice].values
            exog_val = df_product[EXOG_COLS][val_slice].values
            exog_test = df_product[EXOG_COLS][test_slice].values

            exog_scaler = MinMaxScaler()
            exog_train_scaled = exog_scaler.fit_transform(exog_train)
            exog_val_scaled = exog_scaler.transform(exog_val)
            exog_test_scaled = exog_scaler.transform(exog_test)
            
            exog_indices = df_product.columns.get_indexer(EXOG_COLS)
            df_product.iloc[train_slice, exog_indices] = exog_train_scaled
            df_product.iloc[val_slice, exog_indices] = exog_val_scaled
            df_product.iloc[test_slice, exog_indices] = exog_test_scaled
        else:
            exog_scaler = None
            exog_train_scaled = None
            exog_val_scaled = None
            exog_test_scaled = None

        for seed in SEEDS:
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print(f"\n--- RUNNING WITH SEED {seed} ---\n")
            
            grid_search_plots_dir = os.path.join(script_dir, 'grid_search_plots', f'seed_{seed}')
            best_models_seed_dir = os.path.join(script_dir, 'best_models', f'seed_{seed}')
            os.makedirs(grid_search_plots_dir, exist_ok=True)
            os.makedirs(best_models_seed_dir, exist_ok=True)

            all_configs = [{'metric': 'no_emb'}] + grid_configs if USE_EMBEDDINGS else [{'metric': 'no_emb'}]
            
            base_forecast, base_train_losses, base_val_losses = None, None, None
            base_rmse, base_mae, base_bias, base_score, base_pocid = None, None, None, None, None

            for config in all_configs:
                metric = config['metric']
                thresholds = config.get('thresholds', [None])
                percentiles = config.get('percentiles', [None])
                
                results_by_w_s = {}

                if metric == 'no_emb':
                    is_threshold_mode = False
                    iterator = [(None, 15, 1, False, False)]
                else:
                    is_threshold_mode = thresholds is not None and thresholds != [None]
                    params = thresholds if is_threshold_mode else percentiles
                    iterator = itertools.product(params, window_sizes, step_sizes, enable_edges_opts, enable_second_degree_opts)
            
                for param_val, window_sz, step_sz, enable_edges, enable_second_degree in iterator:
                    use_embeddings = (metric != 'no_emb')
                    
                    current_threshold = param_val if use_embeddings and is_threshold_mode else None
                    current_percentile = param_val if use_embeddings and not is_threshold_mode else None

                    key = (param_val, window_sz, step_sz)
                    if key not in results_by_w_s:
                        results_by_w_s[key] = {
                            'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                            'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {},
                            'threshold': None
                        }
                        if metric != 'no_emb' and base_forecast is not None:
                            results_by_w_s[key]['forecasts']["No Embeddings"] = base_forecast
                            results_by_w_s[key]['train_losses']["No Embeddings"] = base_train_losses
                            results_by_w_s[key]['val_losses']["No Embeddings"] = base_val_losses
                            results_by_w_s[key]['rmse']["No Embeddings"] = base_rmse
                            results_by_w_s[key]['mae']["No Embeddings"] = base_mae
                            results_by_w_s[key]['bias']["No Embeddings"] = base_bias
                            results_by_w_s[key]['score']["No Embeddings"] = base_score
                            results_by_w_s[key]['pocid']["No Embeddings"] = base_pocid

                    print(f"\n{'='*60}")
                    if use_embeddings:
                        param_str = f"threshold={current_threshold}" if is_threshold_mode else f"percentile={current_percentile}"
                        print(f"Running MLP Experiment: metric={metric}, {param_str}, window_size={window_sz}, enable_edges={enable_edges}, 2nd_degree={enable_second_degree}")
                    else:
                        print("Running Experiment: BASELINE (no graph embeddings)")
                    print(f"{'='*60}")

                    fixed_threshold = None
                    if use_embeddings:
                        metric_type = infer_metric_type(metric)
                        distance_metrics = ['euclidean','manhattan', 'hamming', 'amplitude_offset', 'slope_consistency', 'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
                        current_df_wide = df_wide_scaled if metric in distance_metrics else df_wide_global
                        compute_func = compute_distances_1vsAll if metric_type == 'distance' else compute_similarities_1vsAll
                        
                        graphs_list, fixed_threshold = neighbourhood_graph(
                            product_id=product_id,
                            df=current_df_wide,
                            metric=metric,
                            metric_type=metric_type,
                            window_size=window_sz,
                            compute_func=compute_func,
                            threshold=current_threshold if is_threshold_mode else None,
                            percentile=current_percentile if not is_threshold_mode else None,
                            step_size=step_sz,
                            cat_labels=cat_labels_dict,
                            residuals=USE_RESIDUALS,
                            enable_edges_within_star=enable_edges,
                            enable_second_degree=enable_second_degree,
                            train_end_idx=val_start_idx
                        )
                        print(f"Resolved graph threshold={current_threshold}: {fixed_threshold}")
                        results_by_w_s[key]['threshold'] = fixed_threshold
                    else:
                        graphs_list = None
                        current_df_wide = df_wide_global

                    cfg = TrainConfig(
                        lookback=lookback_window,
                        horizon=1, 
                        batch_size=batch_size,
                        train_size=train_size,
                        val_size=val_size,
                        lr=LEARNING_RATE,
                        epochs=EPOCHS,
                        device=str(device)
                    )

                    model, _, t_losses, v_losses, best_epoch = train_mlp_forecaster(
                        df=df_product, cfg=cfg, seed=seed, loss_type='mse', 
                        product_id=f"{product_id}_{store_id}", scaler=scaler, target_channel=0, val_ratio=0.2, 
                        hidden_sizes=hidden_sizes, target_col=TARGET_COL, exog_cols=EXOG_COLS, 
                        graphs=graphs_list, test_size=forecast_horizon, sage_in_channels=window_sz + 8
                    )

                    recent_target = val[-lookback_window:].reshape(-1, 1)
                    if exog_val_scaled is not None:
                        recent_exog = exog_val_scaled[-lookback_window:]
                        recent_history = np.column_stack([recent_target, recent_exog])
                    else:
                        recent_history = recent_target

                    start_infer = time.time()
                    
                    if metric == 'no_emb':
                        forecast = recursive_inference_no_graph(
                            model=model,
                            scaler=scaler,
                            recent_history=recent_history,
                            future_exog=exog_test_scaled,
                            target_channel=0,
                            device=str(device),
                        )
                    else:
                        forecast = recursive_inference(
                            model=model,
                            scaler=scaler,
                            recent_history=recent_history,
                            future_exog=exog_test_scaled,
                            target_channel=0,
                            device=str(device),
                            df_wide=current_df_wide,
                            cat_labels=cat_labels_dict,
                            target_id=product_id,
                            metric=metric,
                            fixed_threshold=fixed_threshold,
                            enable_edges_within_star=enable_edges,
                            enable_second_degree=enable_second_degree,
                            past_dates=pd.to_datetime(df_product[DATE_COL][:test_start_idx]).dt.strftime('%Y-%m-%d').values,
                            future_dates=pd.to_datetime(df_product[DATE_COL][test_start_idx:]).dt.strftime('%Y-%m-%d').values,
                            graph_window_size=window_sz
                        )
                    infer_time = time.time() - start_infer
                        
                    valid_mask = ~np.isnan(forecast)
                    valid_test = test[valid_mask]
                    valid_forecast = np.array(forecast)[valid_mask]

                    rmse, mae, bias, score, pocid = None, None, None, None, None
                    if len(valid_test) > 0:
                        rmse = np.sqrt(mean_squared_error(valid_test, valid_forecast))
                        mae = mean_absolute_error(valid_test, valid_forecast)
                        bias = np.mean(valid_forecast - valid_test)
                        score = r2_score(valid_test, valid_forecast)
                        diff_original = valid_test[1:] - valid_test[:-1]
                        diff_pred = valid_forecast[1:] - valid_forecast[:-1]
                        is_positive = (diff_original * diff_pred) > 0
                        pocid = is_positive.sum() / len(is_positive) if len(is_positive) > 0 else 0.0
                        
                    inf_threshold = fixed_threshold if use_embeddings and fixed_threshold is not None else None
                    th_str = f"{inf_threshold:.4f}" if inf_threshold is not None else "N/A"
                    
                    if use_embeddings:
                        param_str_label = f"th:{current_threshold}" if is_threshold_mode else f"pct:{current_percentile} (val:{th_str})"
                        label_name = f"{param_str_label}|w:{window_sz}|st:{step_sz}|e:{enable_edges}|2nd:{enable_second_degree}"
                    else:
                        label_name = "No Embeddings"
                
                    if metric == 'no_emb':
                        base_forecast, base_train_losses, base_val_losses = forecast, t_losses, v_losses
                        base_rmse, base_mae, base_bias = rmse, mae, bias
                        base_score, base_pocid = score, pocid

                    results_by_w_s[key]['forecasts'][label_name] = forecast
                    results_by_w_s[key]['train_losses'][label_name] = t_losses
                    results_by_w_s[key]['val_losses'][label_name] = v_losses
                    results_by_w_s[key]['rmse'][label_name] = rmse
                    results_by_w_s[key]['mae'][label_name] = mae
                    results_by_w_s[key]['bias'][label_name] = bias
                    results_by_w_s[key]['score'][label_name] = score
                    results_by_w_s[key]['pocid'][label_name] = pocid
                
                    print(f"Finished {metric} @ {param_val} -> RMSE: {rmse:.4f}\n")
                    
                    csv_results_path = os.path.join(script_dir, f"{metric}.csv")
                    file_exists = os.path.exists(csv_results_path)
                    with open(csv_results_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(["product_id", "store_id", "seed", "metric", "window_size", "step_size", "threshold", "percentile", "enable_edges", "enable_second_degree", "rmse", "mae", "bias", "r2_score", "pocid"])
                        
                        writer.writerow([product_id, store_id, seed, metric, 15 if metric == 'no_emb' else window_sz, 1 if metric == 'no_emb' else step_sz, current_threshold if use_embeddings else "", current_percentile if use_embeddings else "", enable_edges if use_embeddings else "", enable_second_degree if use_embeddings else "", rmse, mae, bias, score, pocid])

                train_index = df_product[DATE_COL][train_slice].values
                val_index = df_product[DATE_COL][val_slice].values
                test_index = df_product[DATE_COL][test_slice].values

                if metric == 'no_emb':
                    values_str = "no_thresholds"
                    sub_dir = os.path.join(grid_search_plots_dir, 'no_emb', f'window_{15}', f'step_{1}', f'item_{product_id}', values_str)
                    os.makedirs(sub_dir, exist_ok=True)
                    save_plot_path = os.path.join(sub_dir, f"item_{product_id}_store_{store_id}_no_emb_seed_{seed}.html")
                    emb_title = f'Baseline MLP Forecast (No Embeddings | Seed={seed})'
                    
                    if SAVE_PLOTS:
                        plot_results(train, val, test, results_by_w_s[(None, 15, 1)]['forecasts'], train_index, val_index, test_index,
                                     results_by_w_s[(None, 15, 1)]['train_losses'], results_by_w_s[(None, 15, 1)]['val_losses'], metric=metric, embedding_strategy='graph2vec_mlp',
                                     window_size=15, step_size=1, threshold=None, percentile=None,
                                     target_col=TARGET_COL, title=f'{emb_title} (Item={product_id})', seed=seed,
                                     save_path=save_plot_path, rmse=results_by_w_s[(None, 15, 1)]['rmse'], mae=results_by_w_s[(None, 15, 1)]['mae'], 
                                     bias=results_by_w_s[(None, 15, 1)]['bias'], score=results_by_w_s[(None, 15, 1)]['score'], pocid=results_by_w_s[(None, 15, 1)]['pocid'])
                else:
                    metric_type = infer_metric_type(metric)
                    grouped_results = {}
                    for (p, w, s), res_dicts in results_by_w_s.items():
                        group_key = (w, s)
                        if group_key not in grouped_results:
                            grouped_results[group_key] = {
                                'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                                'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {}
                            }
                        grouped_results[group_key]['forecasts'].update(res_dicts['forecasts'])
                        grouped_results[group_key]['train_losses'].update(res_dicts['train_losses'])
                        grouped_results[group_key]['val_losses'].update(res_dicts['val_losses'])
                        grouped_results[group_key]['rmse'].update(res_dicts['rmse'])
                        grouped_results[group_key]['mae'].update(res_dicts['mae'])
                        grouped_results[group_key]['bias'].update(res_dicts['bias'])
                        grouped_results[group_key]['score'].update(res_dicts['score'])
                        grouped_results[group_key]['pocid'].update(res_dicts['pocid'])

                    for (w, s), res_dicts in grouped_results.items():
                        if thresholds is not None and len(thresholds) > 0 and percentiles is None:
                            raw_str = "_".join(map(str, thresholds))
                        else:
                            raw_str = "_".join(map(str, percentiles))
                        
                        values_str = hashlib.md5(raw_str.encode()).hexdigest()[:8]
                        sub_dir = os.path.join(grid_search_plots_dir, metric_type, f'window_{w}', f'step_{s}', f'item_{product_id}', values_str)
                        os.makedirs(sub_dir, exist_ok=True)
                        
                        save_plot_path = os.path.join(sub_dir, f"item_{product_id}_{metric}_seed_{seed}_all_configs.html")
                        emb_title = f'Graph2Vec MLP Forecasts ({metric} | Seed={seed} | W={w} | S={s})'

                        if SAVE_PLOTS:
                            plot_results(train, val, test, res_dicts['forecasts'], train_index, val_index, test_index,
                                         res_dicts['train_losses'], res_dicts['val_losses'], metric=metric, embedding_strategy='graph2vec_mlp',
                                         window_size=w, step_size=s, threshold=None, percentile=None,
                                         target_col=TARGET_COL, title=f'{emb_title} (Item={product_id})', seed=seed,
                                         save_path=save_plot_path, rmse=res_dicts['rmse'], mae=res_dicts['mae'], 
                                         bias=res_dicts['bias'], score=res_dicts['score'], pocid=res_dicts['pocid'])
                                     
    if SAVE_PLOTS:
        print("\nGenerating Correlation Plots across all collected CSVs...")
        for csv_file in os.listdir(script_dir):
            if csv_file.endswith('.csv') and csv_file != 'no_emb.csv' and csv_file != 'mlp_results.csv':
                metric_name = csv_file.replace('.csv', '')
                csv_path = os.path.join(script_dir, csv_file)
                try:
                    res_df = pd.read_csv(csv_path)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    fig.suptitle(f'Threshold vs RMSE and MAE | Metric: {metric_name}', fontsize=16)

                    x_col = 'threshold' if res_df['threshold'].notna().any() else 'percentile'
                    plot_data = res_df.dropna(subset=[x_col, 'rmse', 'mae']).sort_values(by=x_col)
                    
                    if plot_data.empty:
                        continue

                    ax1.plot(plot_data[x_col], plot_data['rmse'], marker='o', linestyle='-', color='b')
                    ax1.set_title(f'{x_col.capitalize()} vs RMSE')
                    ax1.set_xlabel(x_col.capitalize())
                    ax1.set_ylabel('RMSE')
                    ax1.grid(True)

                    ax2.plot(plot_data[x_col], plot_data['mae'], marker='s', linestyle='-', color='r')
                    ax2.set_title(f'{x_col.capitalize()} vs MAE')
                    ax2.set_xlabel(x_col.capitalize())
                    ax2.set_ylabel('MAE')
                    ax2.grid(True)

                    plot_save_path = os.path.join(script_dir, f"{metric_name}_correlation_plot.png")
                    plt.tight_layout()
                    plt.savefig(plot_save_path)
                    plt.close()
                    print(f"Saved correlation plot for {metric_name} at {plot_save_path}")
                except Exception as e:
                    print(f"Failed to generate plot for {csv_file}: {e}")

if __name__ == "__main__":
    main()