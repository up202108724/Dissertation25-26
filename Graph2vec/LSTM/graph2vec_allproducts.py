import os
import random
import sys
import time
import pickle
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths & Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from Baselines.LSTM.lstm import LSTM
from model_utils.utils import generate_exogenous_features, compute_metrics
from Graph2vec.LSTM.graph2vecdataset import TimeSeriesDataset
from Graph2vec.LSTM.train import train_model
from Graph2vec.LSTM.graph2vecinference_adaptativethreshold import graph2vec_inference
from Graph2vec.generate_graph2vecwithadaptativethreshold import load_or_generate_embeddings
from Graph2vec.graph_utils import infer_metric_type
from Graph2vec.experiment_utils import (
    load_panel, build_product_scalers, select_df_wide, split_and_scale,
    new_result_bucket, record_result, merge_result_buckets, config_label, param_grid_hash,
    plot_threshold_correlations, append_result_row,
)
from model_utils.plots import plot_results

# Constants
FULL_DATA_PATH = os.path.join(REPO_ROOT, 'dataset', 'data_andre_classified.feather')
TOP_DATA_PATH = os.path.join(REPO_ROOT, 'dataset', 'top_12500.feather')
DATE_COL = 'date'
TARGET_COL = 'value'


val_size = 31
forecast_horizon = 153
train_size = 761 - val_size - forecast_horizon
lookback_window = 30
BATCH_SIZE = 32
seq_length = lookback_window
DATE_COL = 'date'
TARGET_COL = 'value'
#SEEDS = [42, 1000, 26008, 213626, 907969, 5219788,13451285]  # Add more seeds as needed
SEEDS = [42, 1000, 26008, 555555,213626, 907969, 5219788, 13451285, 23616558, 618626816] # Add more seeds as needed

# Add the products and stores you want to iterate over
PRODUCTS_TO_TEST = None

# EXOG_COLS definition
#EXOG_COLS = []

EXOG_COLS = [
    "day_of_week", "day_of_month", "week_of_year", "week_of_month",
    "month", "quarter", "is_weekend",
    "lag_1", "lag_7",  "lag_30",
    #"rolling_mean_excl_7", "rolling_mean_excl_3", "rolling_mean_excl_5","rolling_mean_excl_15",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_monday", "is_friday",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_pre_holiday_1", "is_pre_holiday_2", "is_pre_holiday_3", "is_pre_holiday_7",
    "is_post_holiday_1", "is_post_holiday_2", "is_post_holiday_3", "is_post_holiday_7",
    "is_bridge_day",
]


# Grid Search Parameters Setup

grid_configs = [
    {'metric': 'spearman', 'thresholds': [0.70]}
]

window_sizes = [30]
step_sizes = [1]
enable_edges_opts = [True]
enable_second_degree_opts = [False]  # We will keep this False for the main analysis, but you can set to True to include second-degree neighbors in the graph construction
USE_RESIDUALS = False
MODEL_TYPE = 'ridge'
EPOCHS = 1000
PATIENCE = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0
SAVE_MODELS = False
SAVE_PLOTS = True
SAVE_EMBEDDINGS = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Load and Preprocess Data (Once for all products)
    full_df, cat_labels_dict, df_wide_global = load_panel(
        FULL_DATA_PATH, EXOG_COLS, generate_exogenous_features, DATE_COL, TARGET_COL,
    )
    # Per-item StandardScalers fitted on the global training window (used for distance metrics)
    product_scalers, df_wide_scaled = build_product_scalers(
        df_wide_global, train_size=455, val_size=154, forecast_horizon=152,
    )

    global PRODUCTS_TO_TEST
    if PRODUCTS_TO_TEST is None:
        top_df = pd.read_feather(TOP_DATA_PATH)
        products_df = top_df[['item_id', 'store_id']].drop_duplicates().reset_index(drop=True)
        PRODUCTS_TO_TEST = list(products_df.itertuples(index=False, name=None))

    for product_id, store_id in PRODUCTS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"PROCESSING PRODUCT {product_id} FOR STORE {store_id}")
        print(f"{'='*80}\n")

        # Filter for the specific product and store
        df = full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)].sort_values(DATE_COL).reset_index(drop=True)



        required_rows = forecast_horizon + val_size + train_size
        if len(df) < required_rows:
            print(f"Skipping Product {product_id} at Store {store_id}: Found {len(df)} rows, but {required_rows} are required for the splits.")
            continue

        sp = split_and_scale(df, TARGET_COL, EXOG_COLS, train_size, val_size, forecast_horizon)
        test_start_idx, val_start_idx = sp.test_start_idx, sp.val_start_idx
        train_slice, val_slice, test_slice = sp.train_slice, sp.val_slice, sp.test_slice
        train, val, test = sp.train, sp.val, sp.test
        scaler, train_scaled, val_scaled, test_scaled = sp.scaler, sp.train_scaled, sp.val_scaled, sp.test_scaled
        exog_scaler, exog_train_scaled = sp.exog_scaler, sp.exog_train_scaled
        exog_val_scaled, exog_test_scaled = sp.exog_val_scaled, sp.exog_test_scaled

        for seed in SEEDS:
            # Set all seeds here
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            print(f"\n--- RUNNING WITH SEED {seed} ---\n")

            # Get the directory where `graph2vec_lstm.py` is located to anchor our save paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            grid_search_plots_dir = os.path.join(script_dir, 'grid_search_plots', f'seed_{seed}')
            best_models_seed_dir = os.path.join(script_dir, 'best_models', f'seed_{seed}')

            os.makedirs(grid_search_plots_dir, exist_ok=True)
            os.makedirs(best_models_seed_dir, exist_ok=True)

            # No-graph baselines are produced by the Baselines/ folder; only graph configs run here.
            for config in grid_configs:
                metric = config['metric']
                thresholds = config.get('thresholds', [None])
                percentiles = config.get('percentiles', [None])

                results_by_w_s = {}

                is_threshold_mode = thresholds is not None and thresholds != [None]
                params = thresholds if is_threshold_mode else percentiles
                iterator = itertools.product(params, window_sizes, step_sizes, enable_edges_opts, enable_second_degree_opts)

                for param_val, window_size, step_size, enable_edges, enable_second_degree in iterator:
                    current_threshold = param_val if is_threshold_mode else None
                    current_percentile = param_val if not is_threshold_mode else None

                    key = (param_val, window_size, step_size)
                    if key not in results_by_w_s:
                        results_by_w_s[key] = new_result_bucket()

                    print(f"\n{'='*60}")
                    param_str = f"threshold={current_threshold}" if is_threshold_mode else f"percentile={current_percentile}"
                    print(f"Running Experiment: metric={metric}, {param_str}, window_size={window_size}, enable_edges={enable_edges}, 2nd_degree={enable_second_degree}")
                    print(f"{'='*60}")

                    metric_type = infer_metric_type(metric)

                    # Use norm-scaled df for distances
                    current_df_wide = select_df_wide(metric, df_wide_global, df_wide_scaled)

                    graph_embeddings, graph2vec_model, csv_path, build_time, emb_time, fixed_threshold = load_or_generate_embeddings(
                        product_id=product_id,
                        metric=metric,
                        metric_type=metric_type,
                        window_size=window_size,
                        step_size=step_size,
                        threshold=current_threshold if is_threshold_mode else None,
                        enable_edges_within_star=enable_edges,
                        enable_second_degree=enable_second_degree,
                        percentile=current_percentile if not is_threshold_mode else None,
                        use_residuals=USE_RESIDUALS,
                        model_type=MODEL_TYPE,
                        seed=seed,
                        train_end_idx=val_start_idx,
                        df=current_df_wide,
                        cat_labels=cat_labels_dict,
                        save_embeddings=SAVE_EMBEDDINGS,
                        output_dir=SCRIPT_DIR,
                    )
                    print(f"Resolved graph threshold={current_threshold}: {fixed_threshold}")
                    print(f"Embedding file: {csv_path}")
                    results_by_w_s[key]['threshold'] = fixed_threshold


                    embedding_dim = graph_embeddings.shape[1] if len(graph_embeddings.shape) > 1 else 1
                    padding = np.zeros((window_size - 1, embedding_dim))
                    aligned_embeddings = np.vstack([padding, graph_embeddings])

                    emb_train = aligned_embeddings[train_slice]
                    emb_val = aligned_embeddings[val_slice]

                    input_size = 1 + (len(EXOG_COLS) if EXOG_COLS else 0) + embedding_dim

                    # Prepend the last seq_length training rows so the val dataset has
                    # enough context to form at least val_size windows.
                    val_target_ctx = np.concatenate([train_scaled[-seq_length:], val_scaled])
                    val_exog_ctx = (
                        np.concatenate([exog_train_scaled[-seq_length:], exog_val_scaled])
                        if EXOG_COLS and len(EXOG_COLS) > 0 else None
                    )
                    val_emb_ctx = np.concatenate([emb_train[-seq_length:], emb_val])

                    train_dataset = TimeSeriesDataset(
                        target_data=train_scaled,
                        exog_data=exog_train_scaled if EXOG_COLS and len(EXOG_COLS) > 0 else None,
                        seq_length=seq_length,
                        embeddings=emb_train,
                        graph_window_size=window_size
                    )
                    use_pin_memory = torch.cuda.is_available()
                    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=use_pin_memory)

                    val_dataset = TimeSeriesDataset(
                        target_data=val_target_ctx,
                        exog_data=val_exog_ctx,
                        seq_length=seq_length,
                        embeddings=val_emb_ctx,
                        graph_window_size=window_size
                    )
                    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=use_pin_memory)

                    # Initialize Model with fixed seed for determinism across iterations!
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)

                    model = LSTM(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)

                    criterion = nn.MSELoss()
                    criterion2 = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE//3)

                    model_dir_label = f"th{current_threshold}" if is_threshold_mode else f"pct{current_percentile}"

                    best_models_dir = os.path.join(best_models_seed_dir, str(window_size), str(step_size), metric, model_dir_label)
                    os.makedirs(best_models_dir, exist_ok=True)

                    if csv_path:
                        # Base the LSTM model name strictly on the embeddings filename
                        csv_basename = os.path.basename(csv_path)
                        base_name = csv_basename.replace('embeddings_', f'best_lstm_{product_id}_')
                        if USE_RESIDUALS:
                            base_name = base_name.replace('.csv', f'_res_{MODEL_TYPE}_seed_{seed}.pth')
                        else:
                            base_name = base_name.replace('.csv', f'_seed_{seed}.pth')

                        hist_name = base_name.replace('.pth', '_history.pkl')
                        best_model_path = os.path.join(best_models_dir, base_name)
                        history_path = os.path.join(best_models_dir, hist_name)
                    else:
                        # Fallback if csv_path is unexpectedly None
                        prefix_star = "" if enable_edges else "star_"
                        if enable_second_degree:
                            prefix_star = "2nddegree_" + prefix_star
                        prefix = f"best_lstm_{prefix_star}{product_id}_{metric}_res_{MODEL_TYPE}" if USE_RESIDUALS else f"best_lstm_{prefix_star}{product_id}_{metric}"
                        param_label = f"th_{current_threshold}" if is_threshold_mode else f"pct_{current_percentile}"
                        best_model_path = os.path.join(best_models_dir, f'{prefix}_{window_size}_{step_size}_{param_label}_seed_{seed}.pth')
                        history_path = os.path.join(best_models_dir, f'{prefix}_{window_size}_{step_size}_{param_label}_seed_{seed}_history.pkl')

                    print(f"Resolved LSTM checkpoint: {best_model_path}")
                    print(f"Resolved LSTM history: {history_path}")

                    if os.path.exists(best_model_path) and os.path.exists(history_path):
                        print(f"Loading existing model from {best_model_path}...")
                        model.load_state_dict(torch.load(best_model_path, weights_only=True))
                        with open(history_path, 'rb') as f:
                            history = pickle.load(f)
                            train_losses = history['train_losses']
                            val_losses = history['val_losses']
                    else:
                        print(f"Model not found. Expected model at {best_model_path}")
                        print(f"Expected history at {history_path}")
                        print("Training new model...")
                        model, train_losses, val_losses, best_epoch, train_time = train_model(
                            seed=seed, epochs=EPOCHS, model=model,
                            train_loader=train_loader, val_loader=val_loader,
                            exog_cols=EXOG_COLS, criterion=criterion, criterion2=criterion2,
                            optimizer=optimizer, device=device,
                            best_model_path=best_model_path if SAVE_MODELS else None, scheduler=scheduler, patience=PATIENCE
                        )

                        if SAVE_MODELS:
                            with open(history_path, 'wb') as f:
                                pickle.dump({
                                    'train_losses': train_losses, 'val_losses': val_losses,
                                    'best_epoch': best_epoch, 'train_time': train_time
                                }, f)

                    # Explicitly load the best saved model before inference to guarantee clean pipeline state
                    if SAVE_MODELS and os.path.exists(best_model_path):
                        print(f"Loading best weights from {best_model_path} for inference...")
                        model.load_state_dict(torch.load(best_model_path))

                    exog_test_data = df[EXOG_COLS][test_slice].values

                    inf_threshold = fixed_threshold

                    print("Running Inference...")

                    forecast, inference_time = graph2vec_inference(
                        metric=metric, window_size=window_size, step_size=step_size,
                        model=model,
                        df=df, df_wide=current_df_wide, cat_labels=cat_labels_dict, date_col=DATE_COL,
                        scaler=scaler, exog_scaler=exog_scaler,
                        test_start_idx=test_start_idx, seq_length=seq_length,
                        forecast_window=forecast_horizon, device=device,
                        item_id=product_id, store_id=store_id, seed=seed,
                        criterion="MSELoss", val_scaled=val_scaled, test_scaled=test_scaled,
                        exog_val_scaled=exog_val_scaled, exog_test_scaled=exog_test_scaled,
                        exog_test_raw=exog_test_data, exog_cols=EXOG_COLS,
                        product_scalers=product_scalers,
                        save_plot_path=None,
                        node_embeddings=aligned_embeddings,
                        graph2vec_model=graph2vec_model,
                        enable_edges_within_star=enable_edges,
                        enable_second_degree=enable_second_degree,
                        percentile=current_percentile if not is_threshold_mode else None,  # Passes percentile mode if used (otherwise None)
                        threshold=inf_threshold,  # Pass the threshold inferred from the percentile
                        create_plots=False  # We will create combined plots later, so disable individual plotting here
                    )

                    valid_mask = ~np.isnan(forecast)
                    valid_test = test[valid_mask]
                    valid_forecast = np.array(forecast)[valid_mask]

                    rmse, mae, bias, score, pocid = None, None, None, None, None
                    try:
                        rmse, mae, bias, score, pocid = compute_metrics(valid_test, valid_forecast)
                    except Exception:
                        if len(valid_test) > 0:
                            rmse = np.sqrt(mean_squared_error(valid_test, valid_forecast))
                            mae = mean_absolute_error(valid_test, valid_forecast)
                            bias = np.mean(valid_forecast - valid_test)
                            score = r2_score(valid_test, valid_forecast)

                    label_name = config_label(
                        is_threshold_mode, current_threshold, current_percentile, inf_threshold,
                        window_size, step_size, enable_edges, enable_second_degree,
                    )

                    record_result(results_by_w_s[key], label_name, dict(
                        forecasts=forecast, train_losses=train_losses, val_losses=val_losses,
                        rmse=rmse, mae=mae, bias=bias, score=score, pocid=pocid,
                    ))

                    print(f"Finished {metric} @ {param_val} -> RMSE: {rmse:.4f}\n")

                    # Append results to a persistent CSV file (all metrics/runs in one file)
                    append_result_row(os.path.join(script_dir, "results.csv"), {
                        "product_id": product_id, "store_id": store_id, "seed": seed, "metric": metric,
                        "window_size": window_size, "step_size": step_size,
                        "threshold": current_threshold, "percentile": current_percentile,
                        "enable_edges": enable_edges, "enable_second_degree": enable_second_degree,
                        "rmse": rmse, "mae": mae, "bias": bias, "r2_score": score, "pocid": pocid,
                    })

                train_index = df[DATE_COL][train_slice].values
                val_index = df[DATE_COL][val_slice].values
                test_index = df[DATE_COL][test_slice].values

                metric_type = infer_metric_type(metric)

                # Group by window and step to combine all thresholds in a single plot
                grouped_results = {}
                for (p, w, s_), res_dicts in results_by_w_s.items():
                    merge_result_buckets(grouped_results.setdefault((w, s_), {}), res_dicts)

                for (w, s_), res_dicts in grouped_results.items():
                    values_str = param_grid_hash(thresholds, percentiles)
                    sub_dir = os.path.join(grid_search_plots_dir, metric_type, f'window_{w}', f'step_{s_}', f'item_{product_id}', values_str)
                    os.makedirs(sub_dir, exist_ok=True)

                    save_plot_path = os.path.join(sub_dir, f"item_{product_id}_{metric}_seed_{seed}_all_configs.html")
                    emb_title = f'Graph2Vec Forecasts ({metric} | Seed={seed} | W={w} | S={s_})'

                    if SAVE_PLOTS:
                        print(f"Saving combined plot to: {os.path.abspath(save_plot_path)}")
                        plot_results(train, val, test, res_dicts['forecasts'], train_index, val_index, test_index,
                                     res_dicts['train_losses'], res_dicts['val_losses'], metric=metric, embedding_strategy='graph2vec',
                                     window_size=w, step_size=s_, threshold=None, percentile=None,
                                     target_col=TARGET_COL, title=f'{emb_title} (Item={product_id})', seed=seed,
                                     save_path=save_plot_path, rmse=res_dicts['rmse'], mae=res_dicts['mae'],
                                     bias=res_dicts['bias'], score=res_dicts['score'], pocid=res_dicts['pocid'])

    # Generate correlation plots at the very end
    if SAVE_PLOTS:
        plot_threshold_correlations(SCRIPT_DIR)

if __name__ == '__main__':
    main()
