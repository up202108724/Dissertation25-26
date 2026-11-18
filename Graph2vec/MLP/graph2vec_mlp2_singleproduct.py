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
    forecast_metrics, new_result_bucket, record_result, merge_result_buckets, config_label,
    param_grid_hash, plot_threshold_correlations,
)
from model_utils.plots import plot_results

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join(REPO_ROOT, 'dataset', 'data_andre_fulfilled.feather')
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
SEEDS = [42,1000,26008,907969]
#seeds = [42,1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]  # Add more seeds as needed
WINDOW_SIZES = [15]     
STEP_SIZES = [1]
ENABLE_EDGES_OPTS = [True]
ENABLE_SECOND_DEGREE_OPTS = [False]  # We will keep this False for the main analysis, but you can set to True to include second-degree neighbors in the graph construction
USE_RESIDUALS = False
SAVE_MODELS = False
SAVE_PLOTS = True
SAVE_EMBEDDINGS = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_EMBEDDINGS_DIM = 16
PRODUCTS_TO_TEST = [
  #(911753,6269)
  (26008, 6269),
  #(907969, 6269),
  #(907967, 6269),
  #(213626, 6269)
]

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    full_df, cat_labels_dict, df_wide_global = load_panel(
        DATA_PATH, EXOG_COLS, generate_exogenous_features, DATE_COL, TARGET_COL,
    )
    # Per-item StandardScalers fitted on the global training window (used for distance metrics)
    product_scalers, df_wide_scaled = build_product_scalers(
        df_wide_global, train_size=train_size, val_size=val_size, forecast_horizon=forecast_horizon,
    )

    os.makedirs('best_models', exist_ok=True)
    os.makedirs('grid_search_plots', exist_ok=True)
    
    for product_id, store_id in PRODUCTS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"PROCESSING PRODUCT {product_id} FOR STORE {store_id}")
        print(f"{'='*80}\n")
        
        df_product = full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)].copy()
        df_product[DATE_COL] = pd.to_datetime(df_product[DATE_COL])
        df_product = df_product.sort_values(DATE_COL).reset_index(drop=True)
        
        sp = split_and_scale(df_product, TARGET_COL, EXOG_COLS, train_size, val_size, forecast_horizon)
        test_start_idx, val_start_idx = sp.test_start_idx, sp.val_start_idx
        train_slice, val_slice, test_slice = sp.train_slice, sp.val_slice, sp.test_slice
        train, val, test = sp.train, sp.val, sp.test
        scaler = sp.scaler
        exog_val_scaled, exog_test_scaled = sp.exog_val_scaled, sp.exog_test_scaled

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

            print(f"\n--- RUNNING WITH SEED {seed} ---\n")
            
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
                iterator = itertools.product(params, WINDOW_SIZES, STEP_SIZES, ENABLE_EDGES_OPTS, ENABLE_SECOND_DEGREE_OPTS)

                for param_val, window_sz, step_sz, enable_edges, enable_second_degree in iterator:
                    current_threshold = param_val if is_threshold_mode else None
                    current_percentile = param_val if not is_threshold_mode else None

                    key = (param_val, window_sz, step_sz)
                    if key not in results_by_w_s:
                        results_by_w_s[key] = new_result_bucket()

                    print(f"\n{'='*60}")
                    param_str = f"threshold={current_threshold}" if is_threshold_mode else f"percentile={current_percentile}"
                    print(f"Running MLP Experiment: metric={metric}, {param_str}, window_size={window_sz}, enable_edges={enable_edges}, 2nd_degree={enable_second_degree}")
                    print(f"{'='*60}")

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
                    print(f"Resolved graph threshold={current_threshold}: {fixed_threshold}")
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
                        graph_window_size=window_sz,
                    )
                    train_time = time.time() - start_train

                    recent_target = val[-lookback_window:].reshape(-1, 1)
                    if exog_val_scaled is not None:
                        recent_exog = exog_val_scaled[-lookback_window:]
                        recent_history = np.column_stack([recent_target, recent_exog])
                    else:
                        recent_history = recent_target

                    recent_embeddings = graph_embeddings[:val_start_idx][-lookback_window:]
                        
                    SAVE_INFERENCE_GRAPHS_PLOTS = True
                    start_infer = time.time()
                    inf_result = recursive_inference(
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
                        future_dates=pd.to_datetime(df_product[DATE_COL][test_start_idx:]).dt.strftime('%Y-%m-%d').values,
                        return_graphs=SAVE_INFERENCE_GRAPHS_PLOTS
                    )
                    infer_time = time.time() - start_infer
                    
                    if SAVE_INFERENCE_GRAPHS_PLOTS:
                        forecast, inference_nx_graphs = inf_result
                        
                        from model_utils.plots import plot_dynamic_graphs
                            
                        graph_plot_dir = os.path.join(script_dir, 'graph_infered_plots', str(product_id), f'seed_{seed}')
                        os.makedirs(graph_plot_dir, exist_ok=True)
                        print(f"\nSaving inference graphs plots for product {product_id} (Seed: {seed}) ...")
                        plot_dynamic_graphs(
                            graphs=inference_nx_graphs,
                            product_id=product_id,
                            metric=metric,
                            plot_dir=graph_plot_dir,
                            residuals=USE_RESIDUALS,
                            enable_edges_within_star=enable_edges,
                            enable_second_degree=enable_second_degree,
                            num_plots=None,
                            window_size=window_sz,
                            step_size=step_sz,
                            threshold=fixed_threshold,
                            percentile=current_percentile
                        )
                    else:
                        forecast = inf_result if not (type(inf_result) is tuple) else inf_result[0]
                        
                    valid_mask = ~np.isnan(forecast)
                    valid_test = test[valid_mask]
                    valid_forecast = np.array(forecast)[valid_mask]

                    rmse, mae, bias, score, pocid = forecast_metrics(valid_test, valid_forecast)

                    label_name = config_label(
                        is_threshold_mode, current_threshold, current_percentile, fixed_threshold,
                        window_sz, step_sz, enable_edges, enable_second_degree,
                    )
                    record_result(results_by_w_s[key], label_name, dict(
                        forecasts=forecast, train_losses=t_losses, val_losses=v_losses,
                        rmse=rmse, mae=mae, bias=bias, score=score, pocid=pocid,
                    ))
                
                    print(f"Finished {metric} @ {param_val} -> RMSE: {rmse:.4f}\n")
                    
                    csv_results_path = os.path.join(script_dir, f"{metric}.csv")
                    file_exists = os.path.exists(csv_results_path)
                    with open(csv_results_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(["product_id", "store_id", "seed", "metric", "window_size", "step_size", "threshold", "percentile", "enable_edges", "enable_second_degree", "rmse", "mae", "bias", "r2_score", "pocid", "train_time", "infer_time"])
                        
                        writer.writerow([product_id, store_id, seed, metric, window_sz, step_sz, current_threshold, current_percentile, enable_edges, enable_second_degree, rmse, mae, bias, score, pocid, round(train_time, 4), round(infer_time, 4)])

                train_index = df_product[DATE_COL][train_slice].values
                val_index = df_product[DATE_COL][val_slice].values
                test_index = df_product[DATE_COL][test_slice].values

                metric_type = infer_metric_type(metric)
                grouped_results = {}
                for (p, w, s), res_dicts in results_by_w_s.items():
                    merge_result_buckets(grouped_results.setdefault((w, s), {}), res_dicts)

                for (w, s), res_dicts in grouped_results.items():
                    values_str = param_grid_hash(thresholds, percentiles)
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
        plot_threshold_correlations(script_dir, skip=('no_emb.csv', 'mlp_results.csv'))

if __name__ == "__main__":
    main()