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
sys.path.append(os.path.join(script_dir, '..', 'LSTM'))

from utils import generate_exogenous_features
from train import TrainConfig, train_mlp_forecaster
from graph2vecinference import recursive_inference
from plots import plot_results

from generate_graph2vecwithadaptativethreshold import load_or_generate_embeddings, infer_metric_type
import itertools
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FULL_DATA_PATH = os.path.join(script_dir, '..', '..','..', 'dataset', 'data_andre_classified.feather')
TOP_DATA_PATH = os.path.join(script_dir, '..', '..','..', 'dataset', 'top_12500.feather')
DATE_COL = 'date'
TARGET_COL = 'value'


val_size = 30
forecast_horizon = 153
train_size = 761 - val_size - forecast_horizon
lookback_window = 30

EXOG_COLS = [
    # Cyclical Calendar Features 
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "dom_sin", "dom_cos", "wom_sin", "wom_cos", "month_sin", "month_cos", "quarter_sin", "quarter_cos", "woy_sin", "woy_cos",
 
    # Structural boundaries 
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
   
    # Trend Hint 
    "rolling_mean_excl_7",
   
 
    # Holidays & Events (Crucial)
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_bridge_day"
]
grid_configs = [

    {'metric': 'spearman', 'thresholds': [0.75,0.82,0.85,0.88,0.91]},
]

# Training hyperparameters
BATCH_SIZE = 32
HIDDEN_SIZES = (128, 64)
MLP_MODEL_TYPE = "tdmlp"
DROPOUT = 0.2
EPOCHS = 1000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MODEL_TYPE = 'ridge'
LOSS_TYPE = 'mse'
PATIENCE = 150
##########################
SEEDS = [42]
#seeds = [42,1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]  # Add more seeds as needed
WINDOW_SIZES = [15]     
STEP_SIZES = [1]
ENABLE_EDGES_OPTS = [True]
ENABLE_SECOND_DEGREE_OPTS = [False]  # We will keep this False for the main analysis, but you can set to True to include second-degree neighbors in the graph construction
USE_RESIDUALS = False
SAVE_MODELS = False
SAVE_PLOTS = True
USE_EMBEDDINGS = True
SAVE_EMBEDDINGS = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_EMBEDDINGS_DIM = 16
# TARGET_CATEGORIES is no longer used since we test everything in the feather file

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    experiment_start = time.time()
    df = pd.read_feather(FULL_DATA_PATH)
    
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

    # ── Select test products from the top dataset ────────────────────────────
    top_df = pd.read_feather(TOP_DATA_PATH)
    products_df = (
        top_df[['item_id', 'store_id']]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    PRODUCTS_TO_TEST = list(products_df.itertuples(index=False, name=None))

    results_csv = os.path.join(script_dir, "graph2vec_mlp2_results.csv")
    done_set = set()
    if os.path.exists(results_csv):
        done_df = pd.read_csv(results_csv, dtype=str)
        # Using string matching to be safe with float representations
        done_df['threshold'] = done_df['threshold'].fillna('').astype(str)
        done_df['percentile'] = done_df['percentile'].fillna('').astype(str)
        for _, row in done_df.iterrows():
            # key: (product_id, store_id, seed, metric, threshold)
            done_set.add((str(row['item_id']), str(row['store_id']), str(row['seed']), str(row['metric']), str(row['threshold'])))
        print(f"Resuming: {len(done_set)} experiments already completed.")

    for product_id, store_id in PRODUCTS_TO_TEST:
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

            best_models_seed_dir = os.path.join(script_dir, 'best_models', f'seed_{seed}')
            os.makedirs(best_models_seed_dir, exist_ok=True)

            all_configs = [{'metric': 'no_emb'}] + grid_configs if USE_EMBEDDINGS else [{'metric': 'no_emb'}]
            
            base_forecast, base_train_losses, base_val_losses = None, None, None
            base_rmse, base_mae, base_bias, base_score, base_pocid = None, None, None, None, None

            all_forecasts_for_plot = {}
            all_train_losses_for_plot = {}
            all_val_losses_for_plot = {}
            all_rmse_for_plot = {}
            all_mae_for_plot = {}
            all_bias_for_plot = {}
            all_score_for_plot = {}
            all_pocid_for_plot = {}

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
                    iterator = itertools.product(params, WINDOW_SIZES, STEP_SIZES, ENABLE_EDGES_OPTS, ENABLE_SECOND_DEGREE_OPTS)
            
                for param_val, window_sz, step_sz, enable_edges, enable_second_degree in iterator:
                    use_embeddings = (metric != 'no_emb')
                    
                    current_threshold = param_val if use_embeddings and is_threshold_mode else None
                    current_percentile = param_val if use_embeddings and not is_threshold_mode else None

                    # Resumable script check
                    exp_th_str = str(current_threshold) if use_embeddings and is_threshold_mode else ""
                    if (str(product_id), str(store_id), str(seed), str(metric), exp_th_str) in done_set:
                        print(f"Skipping already completed experiment: Item {product_id}, Store {store_id}, Seed {seed}, Metric {metric}, Threshold {exp_th_str}")
                        continue

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

                    fixed_threshold = None
                    if use_embeddings:
                        metric_type = infer_metric_type(metric)
                        distance_metrics = ['euclidean','manhattan', 'hamming', 'amplitude_offset', 'slope_consistency', 'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
                        current_df_wide = df_wide_scaled if metric in distance_metrics else df_wide_global
                        
                        graph_embeddings, graph2vec_model, csv_path, build_time, emb_time, fixed_threshold = load_or_generate_embeddings(
                            product_id=product_id,
                            metric=metric,
                            metric_type=metric_type,
                            window_size=window_sz,
                            step_size=step_sz,
                            threshold=current_threshold if is_threshold_mode else None,
                            dimensions=GRAPH_EMBEDDINGS_DIM,
                            enable_edges_within_star=enable_edges,
                            enable_second_degree=enable_second_degree,
                            percentile=current_percentile if not is_threshold_mode else None,
                            use_residuals=USE_RESIDUALS,
                            model_type=MODEL_TYPE,
                            seed=seed,
                            train_end_idx=val_start_idx,
                            df=current_df_wide,
                            cat_labels=cat_labels_dict,
                            save_embeddings=SAVE_EMBEDDINGS
                        )
                        results_by_w_s[key]['threshold'] = fixed_threshold
                    else:
                        graph_embeddings = None
                        graph2vec_model = None
                        current_df_wide = df_wide_global

                    cfg = TrainConfig(
                        lookback=lookback_window,
                        horizon=1,
                        batch_size=BATCH_SIZE,
                        train_size=train_size,
                        val_size=val_size,
                        lr=LEARNING_RATE,
                        dropout=DROPOUT,
                        epochs=EPOCHS,
                        patience=PATIENCE,
                        hidden_sizes=HIDDEN_SIZES,
                        weight_decay=WEIGHT_DECAY,
                        model_type=MLP_MODEL_TYPE,
                        device=str(DEVICE)
                    )

                    start_train = time.time()
                    model, _, t_losses, v_losses, best_epoch = train_mlp_forecaster(
                        df=df_product, cfg=cfg, seed=seed, loss_type=LOSS_TYPE,
                        product_id=f"{product_id}_{store_id}", scaler=scaler, target_channel=0,
                        target_col=TARGET_COL, exog_cols=EXOG_COLS,
                        graph_embeddings=graph_embeddings, test_size=forecast_horizon
                    )
                    train_time = time.time() - start_train

                    recent_target = val[-lookback_window:].reshape(-1, 1)
                    if exog_val_scaled is not None:
                        recent_exog = exog_val_scaled[-lookback_window:]
                        recent_history = np.column_stack([recent_target, recent_exog])
                    else:
                        recent_history = recent_target

                    recent_embeddings = graph_embeddings[:val_start_idx][-lookback_window:] if graph_embeddings is not None else None
                        
                    start_infer = time.time()
                    forecast = recursive_inference(
                        model=model,
                        scaler=scaler,
                        recent_history=recent_history,
                        future_exog=exog_test_scaled,
                        target_channel=0,
                        device=str(DEVICE),
                        graph2vec_model=graph2vec_model,
                        graph_window_size=window_sz,
                        recent_embeddings=recent_embeddings,
                        df_wide=current_df_wide,
                        cat_labels=cat_labels_dict,
                        target_id=product_id,
                        metric=metric,
                        fixed_threshold=fixed_threshold,
                        enable_edges_within_star=enable_edges,
                        enable_second_degree=enable_second_degree,
                        past_dates=pd.to_datetime(df_product[DATE_COL][:test_start_idx]).dt.strftime('%Y-%m-%d').values,
                        future_dates=pd.to_datetime(df_product[DATE_COL][test_start_idx:]).dt.strftime('%Y-%m-%d').values
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

                    all_forecasts_for_plot[label_name] = forecast
                    all_train_losses_for_plot[label_name] = t_losses
                    all_val_losses_for_plot[label_name] = v_losses
                    all_rmse_for_plot[label_name] = rmse
                    all_mae_for_plot[label_name] = mae
                    all_bias_for_plot[label_name] = bias
                    all_score_for_plot[label_name] = score
                    all_pocid_for_plot[label_name] = pocid
                
                    csv_results_path = os.path.join(script_dir, "graph2vec_mlp2_results.csv")
                    file_exists = os.path.exists(csv_results_path)
                    with open(csv_results_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(["item_id", "store_id", "seed", "metric", "window_size", "step_size", "threshold", "percentile", "enable_edges", "enable_second_degree", "rmse", "mae", "bias", "r2_score", "pocid", "train_time", "inference_time", "best_epoch"])
                        
                        writer.writerow([product_id, store_id, seed, metric, 15 if metric == 'no_emb' else window_sz, 1 if metric == 'no_emb' else step_sz, current_threshold if use_embeddings else "", current_percentile if use_embeddings else "", enable_edges if use_embeddings else "", enable_second_degree if use_embeddings else "", rmse, mae, bias, score, pocid, round(train_time, 4), round(infer_time, 4), best_epoch])

            if SAVE_PLOTS:
                plot_dir = os.path.join(script_dir, f'grid_search_plots/seed_{seed}/{LOSS_TYPE}')
                os.makedirs(plot_dir, exist_ok=True)
                
                plot_path = os.path.join(plot_dir, f'item{product_id}_store{store_id}_comparison.html')
                
                train_index = df_product[DATE_COL][train_slice].values
                val_index = df_product[DATE_COL][val_slice].values
                test_index = df_product[DATE_COL][test_slice].values
                
                plot_results(train, val, test, all_forecasts_for_plot, 
                             train_index, val_index, test_index,
                             all_train_losses_for_plot, all_val_losses_for_plot,
                             target_col=TARGET_COL, title=f'TDMLP Forecast - Item {product_id} Store {store_id} (Seed {seed})',
                             save_path=plot_path,
                             rmse=all_rmse_for_plot, mae=all_mae_for_plot, bias=all_bias_for_plot,
                             score=all_score_for_plot, pocid=all_pocid_for_plot, df_full=df_product)

    # ------------------------------------------------------------------
    # Timing summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - experiment_start
    csv_results_path = os.path.join(script_dir, "graph2vec_mlp2_results.csv")
    with open(csv_results_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["TOTAL_EXPERIMENT_TIME_SECONDS", round(total_elapsed, 2)] + [""] * 16)
        
    print(f"\nTotal experiment time: {total_elapsed/60:.2f} min → saved to {csv_results_path}")

if __name__ == "__main__":
    main()