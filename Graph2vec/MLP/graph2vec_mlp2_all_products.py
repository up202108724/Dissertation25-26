import sys
import os
import time
import itertools
import csv
import pandas as pd
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(script_dir, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from model_utils.utils import generate_exogenous_features
from Graph2vec.MLP.train import TrainConfig, train_mlp_forecaster
from Graph2vec.MLP.graph2vecmlpinference import recursive_inference
from Graph2vec.generate_graph2vecwithadaptativethreshold import load_or_generate_embeddings
from Graph2vec.graph_utils import infer_metric_type
from Graph2vec.experiment_utils import (
    load_panel, build_product_scalers, select_df_wide, split_and_scale,
    forecast_metrics, new_result_bucket, record_result, config_label, append_result_row,
)
from model_utils.plots import plot_results

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FULL_DATA_PATH = os.path.join(REPO_ROOT, 'dataset', 'data_andre_classified.feather')
TOP_DATA_PATH = os.path.join(REPO_ROOT, 'dataset', 'top_12500.feather')
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

    {'metric': 'spearman', 'thresholds': [0.70]},
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
SEEDS = [42, 1000, 26008, 555555,213626, 907969, 5219788, 13451285, 23616558, 618626816] # Add more seeds as needed
WINDOW_SIZES = [30]     
STEP_SIZES = [1]
ENABLE_EDGES_OPTS = [True]
ENABLE_SECOND_DEGREE_OPTS = [False]  # We will keep this False for the main analysis, but you can set to True to include second-degree neighbors in the graph construction
USE_RESIDUALS = False
SAVE_MODELS = False
SAVE_PLOTS = True
SAVE_EMBEDDINGS = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_EMBEDDINGS_DIM = 16
# TARGET_CATEGORIES is no longer used since we test everything in the feather file

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    experiment_start = time.time()
    full_df, cat_labels_dict, df_wide_global = load_panel(
        FULL_DATA_PATH, EXOG_COLS, generate_exogenous_features, DATE_COL, TARGET_COL,
    )
    # Per-item StandardScalers fitted on the global training window (used for distance metrics)
    product_scalers, df_wide_scaled = build_product_scalers(
        df_wide_global, train_size=train_size, val_size=val_size, forecast_horizon=forecast_horizon,
    )

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
        # Older files also hold no_emb baseline and ablate_z=True rows; only full graph runs count as done.
        done_df = done_df[done_df['metric'] != 'no_emb']
        if 'ablate_z' in done_df.columns:
            done_df = done_df[done_df['ablate_z'].fillna('False').astype(str) != 'True']
        for _, row in done_df.iterrows():
            # key: (product_id, store_id, seed, metric, threshold)
            done_set.add((str(row['item_id']), str(row['store_id']), str(row['seed']), str(row['metric']), str(row['threshold'])))
        print(f"Resuming: {len(done_set)} experiments already completed.")

    for product_id, store_id in PRODUCTS_TO_TEST:
        df_product = full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)].copy()
        df_product[DATE_COL] = pd.to_datetime(df_product[DATE_COL])
        df_product = df_product.sort_values(DATE_COL).reset_index(drop=True)
        
        sp = split_and_scale(df_product, TARGET_COL, EXOG_COLS, train_size, val_size, forecast_horizon)
        test_start_idx, val_start_idx = sp.test_start_idx, sp.val_start_idx
        train_slice, val_slice, test_slice = sp.train_slice, sp.val_slice, sp.test_slice
        train, val, test = sp.train, sp.val, sp.test
        scaler = sp.scaler
        exog_scaler, exog_val_scaled, exog_test_scaled = sp.exog_scaler, sp.exog_val_scaled, sp.exog_test_scaled

        if EXOG_COLS:
            # train_mlp_forecaster reads exogenous columns from df_product, so write the scaled values back
            exog_indices = df_product.columns.get_indexer(EXOG_COLS)
            df_product.iloc[train_slice, exog_indices] = sp.exog_train_scaled
            df_product.iloc[val_slice, exog_indices] = exog_val_scaled
            df_product.iloc[test_slice, exog_indices] = exog_test_scaled

        for seed in SEEDS:
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            best_models_seed_dir = os.path.join(script_dir, 'best_models', f'seed_{seed}')
            os.makedirs(best_models_seed_dir, exist_ok=True)

            plot_bucket = new_result_bucket()  # every config of this seed, overlaid in one plot

            # No-graph baselines are produced by the Baselines/ folder; only graph configs run here.
            for config in grid_configs:
                metric = config['metric']
                thresholds = config.get('thresholds', [None])
                percentiles = config.get('percentiles', [None])

                results_by_w_s = {}

                is_threshold_mode = thresholds is not None and thresholds != [None]
                params = thresholds if is_threshold_mode else percentiles
                iterator = itertools.product(params, WINDOW_SIZES, STEP_SIZES, ENABLE_EDGES_OPTS, ENABLE_SECOND_DEGREE_OPTS)

                for param_val, window_sz, step_sz, enable_edges, enable_second_degree in iterator:
                    current_threshold = param_val if is_threshold_mode else None
                    current_percentile = param_val if not is_threshold_mode else None

                    # Resumable script check
                    exp_th_str = str(current_threshold) if is_threshold_mode else ""
                    if (str(product_id), str(store_id), str(seed), str(metric), exp_th_str) in done_set:
                        print(f"Skipping already completed experiment: Item {product_id}, Store {store_id}, Seed {seed}, Metric {metric}, Threshold {exp_th_str}")
                        continue

                    key = (param_val, window_sz, step_sz)
                    if key not in results_by_w_s:
                        results_by_w_s[key] = new_result_bucket()

                    metric_type = infer_metric_type(metric)
                    current_df_wide = select_df_wide(metric, df_wide_global, df_wide_scaled)

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
                        save_embeddings=SAVE_EMBEDDINGS,
                        output_dir=script_dir,
                    )
                    results_by_w_s[key]['threshold'] = fixed_threshold

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
                        graph_embeddings=graph_embeddings, test_size=forecast_horizon,
                        graph_window_size=window_sz
                    )
                    train_time = time.time() - start_train

                    recent_target = val[-lookback_window:].reshape(-1, 1)
                    if exog_val_scaled is not None:
                        recent_exog = exog_val_scaled[-lookback_window:]
                        recent_history = np.column_stack([recent_target, recent_exog])
                    else:
                        recent_history = recent_target

                    recent_embeddings = graph_embeddings[:val_start_idx][-lookback_window:]
                        
                    start_infer = time.time()
                    forecast = recursive_inference(
                        model=model,
                        scaler=scaler,
                        recent_history=recent_history,
                        future_exog=exog_test_scaled,
                        target_channel=0,
                        device=str(DEVICE),
                        exog_cols=EXOG_COLS,
                        exog_scaler=exog_scaler,
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

                    rmse, mae, bias, score, pocid = forecast_metrics(valid_test, valid_forecast)

                    label_name = config_label(
                        is_threshold_mode, current_threshold, current_percentile, fixed_threshold,
                        window_sz, step_sz, enable_edges, enable_second_degree,
                    )

                    run_result = dict(
                        forecasts=forecast, train_losses=t_losses, val_losses=v_losses,
                        rmse=rmse, mae=mae, bias=bias, score=score, pocid=pocid,
                    )
                    record_result(results_by_w_s[key], label_name, run_result)
                    record_result(plot_bucket, label_name, run_result)
                
                    # Existing results files keep their header (incl. the old ablate_z column, left empty)
                    append_result_row(results_csv, {
                        "item_id": product_id, "store_id": store_id, "seed": seed, "metric": metric,
                        "window_size": window_sz, "step_size": step_sz,
                        "threshold": current_threshold, "percentile": current_percentile,
                        "enable_edges": enable_edges, "enable_second_degree": enable_second_degree,
                        "rmse": rmse, "mae": mae, "bias": bias, "r2_score": score, "pocid": pocid,
                        "train_time": round(train_time, 4), "inference_time": round(infer_time, 4),
                        "best_epoch": best_epoch,
                    })

            if SAVE_PLOTS:
                plot_dir = os.path.join(script_dir, f'grid_search_plots/seed_{seed}/{LOSS_TYPE}')
                os.makedirs(plot_dir, exist_ok=True)
                
                plot_path = os.path.join(plot_dir, f'item{product_id}_store{store_id}_comparison.html')
                
                train_index = df_product[DATE_COL][train_slice].values
                val_index = df_product[DATE_COL][val_slice].values
                test_index = df_product[DATE_COL][test_slice].values
                
                plot_results(train, val, test, plot_bucket['forecasts'],
                             train_index, val_index, test_index,
                             plot_bucket['train_losses'], plot_bucket['val_losses'],
                             target_col=TARGET_COL, title=f'TDMLP Forecast - Item {product_id} Store {store_id} (Seed {seed})',
                             save_path=plot_path,
                             rmse=plot_bucket['rmse'], mae=plot_bucket['mae'], bias=plot_bucket['bias'],
                             score=plot_bucket['score'], pocid=plot_bucket['pocid'], df_full=df_product)

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