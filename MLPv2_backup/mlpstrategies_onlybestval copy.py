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

script_dir = os.path.dirname(os.path.abspath(__file__))
# Add LSTM directory to path to reuse training scripts, datasets and plots
sys.path.append(os.path.join(script_dir, '..', 'LSTM'))

from plots import plot_results
from utils import generate_exogenous_features, ExogenousScaler
from train import TrainConfig, train_mlp_forecaster, train_model_best_train_loss, train_model_combined, train_model_expanding_window, train_model_sliding_window
from inference import recursive_inference_dynamic_exog
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join(script_dir, '..', 'dataset', 'data_andre.feather')
DATE_COL = 'date'
TARGET_COL = 'value'

train_size = 455
val_size = 154
forecast_horizon = 152
lookback_window = 30
expected_series_days = train_size + val_size + forecast_horizon

# Trimmed: removed redundant ordinal duplicates of cyclical encodings
# (day_of_week vs dow_*, month vs month_*, is_monday/is_friday vs dow_*).
# Rolling means use the leak-safe "_excl_" variant — see utils.py.
EXOG_COLS = [
    "dom_sin","dom_cos", "wom_sin", "wom_cos",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend",
    "lag_1", "lag_7", "lag_14","lag_28",
    "rolling_mean_excl_3", 
    #"rolling_mean_excl_5",
    "rolling_mean_excl_7", "rolling_mean_excl_14", "rolling_mean_excl_28",
    "month_sin", "month_cos", "quarter",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_bridge_day",
]

batch_size = 32
hidden_sizes = (256,64)
dropout = 0.2
EPOCHS = 1000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
#seeds = [42]
seeds = [42, 1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]  # Full seed grid

loss_type = 'huber'

# Toggle per-product HTML plot generation. Disable to speed up large grid runs.
ENABLE_PLOTS = False


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

    # Enforce a complete 761-day daily series for each (item_id, store_id).
    # Any missing day is imputed as 0 in the target.
    full_date_index = pd.date_range(df[DATE_COL].min(), periods=expected_series_days, freq='D')
    all_products = df[['item_id', 'store_id']].drop_duplicates().values
    completed_series = []
    for item_id, store_id in all_products:
        series_df = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)][[DATE_COL, TARGET_COL]].copy()
        series_df = series_df.groupby(DATE_COL, as_index=False)[TARGET_COL].sum()
        series_df = series_df.set_index(DATE_COL).reindex(full_date_index)
        series_df.index.name = DATE_COL
        series_df[TARGET_COL] = series_df[TARGET_COL].fillna(0.0)
        series_df['item_id'] = item_id
        series_df['store_id'] = store_id
        completed_series.append(series_df.reset_index())

    df = pd.concat(completed_series, ignore_index=True)
    df = df.sort_values([DATE_COL, 'item_id', 'store_id']).reset_index(drop=True)
    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS)

    products = df[['item_id', 'store_id']].drop_duplicates().values
    strategies= ['best_val']
    #strategies = ['best_val', 'best_train_early_val', 'combined', 'expanding_window', 'sliding_window']
    all_results = []
    per_seed_csv_paths = []

    results_root = os.path.join(script_dir, 'MLP_Results')
    os.makedirs(results_root, exist_ok=True)
    os.makedirs('best_models', exist_ok=True)
    os.makedirs('grid_search_plots', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #criterion = nn.MSELoss()
    #criterion2 = nn.MSELoss()

    for seed in seeds:
        results = []
        seed_results_dir = os.path.join(results_root, f'seed_{seed}', loss_type)
        os.makedirs(seed_results_dir, exist_ok=True)
        seed_csv_path = os.path.join(seed_results_dir, 'results.csv')
        for item_id, store_id in products:
            df_product = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].copy()
            df_product[DATE_COL] = pd.to_datetime(df_product[DATE_COL])
            df_product = df_product.sort_values(DATE_COL).reset_index(drop=True)

            if len(df_product) != expected_series_days:
                print(f"Skipping item {item_id} / store {store_id}: expected {expected_series_days} rows, got {len(df_product)}")
                continue
            
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

            # Keep an UNSCALED copy of the exog block — needed by the dynamic
            # inference loop to recompute lag/rolling features from predictions.
            exog_train_unscaled = df_product[EXOG_COLS].iloc[train_slice].copy()
            exog_val_unscaled = df_product[EXOG_COLS].iloc[val_slice].copy()
            exog_test_unscaled = df_product[EXOG_COLS].iloc[test_slice].copy()

            if len(EXOG_COLS) > 0:
                # Type-aware scaler: pass-through for binary/cyclical, MinMax for continuous.
                exog_scaler = ExogenousScaler(continuous_strategy='minmax')
                exog_scaler.fit(exog_train_unscaled, EXOG_COLS)

                exog_train_scaled = exog_scaler.transform(exog_train_unscaled.copy(), EXOG_COLS)
                exog_val_scaled = exog_scaler.transform(exog_val_unscaled.copy(), EXOG_COLS)
                exog_test_scaled = exog_scaler.transform(exog_test_unscaled.copy(), EXOG_COLS)

                # Write the SCALED exog values back into df_product so train.py
                # (which reads df_product directly) sees the scaled features.
                exog_col_idx = df_product.columns.get_indexer(EXOG_COLS)
                df_product.iloc[train_slice, exog_col_idx] = exog_train_scaled[EXOG_COLS].values
                df_product.iloc[val_slice, exog_col_idx] = exog_val_scaled[EXOG_COLS].values
                df_product.iloc[test_slice, exog_col_idx] = exog_test_scaled[EXOG_COLS].values
            else:
                exog_scaler = None
            
            input_size = 1 + len(EXOG_COLS)
            
            # Dataloaders and Dataset extraction deleted since the updated `train.py` functions use df & TrainConfig directly.

            strategy_forecasts = {}
            strategy_train_losses = {}
            strategy_val_losses = {}
            strategy_rmses = {}
            strategy_maes = {}
            strategy_biases = {}
            strategy_scores = {}
            strategy_pocids = {}

            train_index = df_product[DATE_COL][train_slice].values
            val_index = df_product[DATE_COL][val_slice].values
            test_index = df_product[DATE_COL][test_slice].values

            for strategy in strategies:
                print(f"\\n--- Strategy: {strategy}, Seed: {seed}, Item: {item_id}, Store: {store_id} ---")
                
                # Removed manual MLP instantiation here because it's now created inside the train script

                model_dir = f'best_models/seed_{seed}/{loss_type}/{strategy}'
                os.makedirs(model_dir, exist_ok=True)
                model_path = f'{model_dir}/mlp_item{item_id}_store{store_id}.pth'
                
                
                # Prepare global config for this strategy
                cfg = TrainConfig(
                    lookback=lookback_window,
                    horizon=1, 
                    batch_size=batch_size,
                    train_size=train_size,
                    val_size=val_size,
                    lr=LEARNING_RATE,
                    epochs=EPOCHS,
                    weight_decay=WEIGHT_DECAY,
                    device=str(device)
                )

                if strategy == 'best_val':
                    model, _, t_losses, v_losses, best_epoch = train_mlp_forecaster(
                        df=df_product, cfg=cfg, seed=seed, loss_type='mse', 
                        product_id=f"{item_id}_{store_id}", scaler=scaler, target_channel=0, val_ratio=0.2, 
                        hidden_sizes=hidden_sizes, target_col=TARGET_COL, exog_cols=EXOG_COLS, test_size=forecast_horizon)
                    train_time = 0.0 # Time block removed from baseline signature
                elif strategy == 'best_train_early_val':
                    model, _, t_losses, v_losses, best_epoch = train_model_best_train_loss(
                        df=df_product, cfg=cfg, seed=seed, loss_type='mse', 
                        product_id=f"{item_id}_{store_id}", scaler=scaler, target_channel=0, val_ratio=0.2, 
                        hidden_sizes=hidden_sizes, target_col=TARGET_COL, exog_cols=EXOG_COLS, test_size=forecast_horizon)
                    train_time = 0.0
                elif strategy == 'combined':
                    model, _, _, v_losses, optimal_epoch = train_mlp_forecaster(
                        df=df_product, cfg=cfg, seed=seed, loss_type='mse', 
                        product_id=f"{item_id}_{store_id}_temp", scaler=scaler, target_channel=0, val_ratio=0.2, 
                        hidden_sizes=hidden_sizes, target_col=TARGET_COL, exog_cols=EXOG_COLS, test_size=forecast_horizon)
                    
                    cfg.epochs = optimal_epoch if optimal_epoch > 0 else EPOCHS
                    model, _, t_losses, train_time = train_model_combined(
                         df=df_product, cfg=cfg, seed=seed, loss_type='mse', 
                         product_id=f"{item_id}_{store_id}", scaler=scaler, target_channel=0, val_ratio=0.2, 
                         hidden_sizes=hidden_sizes, target_col=TARGET_COL, exog_cols=EXOG_COLS, test_size=forecast_horizon)
                    v_losses = []
                    best_epoch = optimal_epoch
                elif strategy == 'expanding_window':
                    model, _, t_losses, v_losses, best_epoch = train_model_expanding_window(
                        df=df_product, cfg=cfg, seed=seed, loss_type='mse', 
                        product_id=f"{item_id}_{store_id}", scaler=scaler, target_channel=0, val_ratio=0.2, 
                        hidden_sizes=hidden_sizes, target_col=TARGET_COL, exog_cols=EXOG_COLS, test_size=forecast_horizon)
                    train_time = 0.0
                elif strategy == 'sliding_window':
                    model, _, t_losses, v_losses, best_epoch = train_model_sliding_window(
                        df=df_product, cfg=cfg, seed=seed, loss_type='mse', 
                        product_id=f"{item_id}_{store_id}", scaler=scaler, target_channel=0, val_ratio=0.2, 
                        hidden_sizes=hidden_sizes, target_col=TARGET_COL, exog_cols=EXOG_COLS, test_size=forecast_horizon)
                    train_time = 0.0

                strategy_train_losses[strategy] = t_losses
                strategy_val_losses[strategy] = v_losses

                # Load best model
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path))
                
                # Leak-safe recursive inference:
                #   - history uses the last `lookback` UNSCALED target & exog values
                #     (their lag/rolling cols were generated from ground-truth
                #      train+val data, which is allowed),
                #   - future_exog is passed UNSCALED so the inference loop can
                #     OVERWRITE lag_*/rolling_mean_* per step using the running
                #     prediction buffer (no peek at the true test target).
                recent_target_unscaled = val[-lookback_window:].astype(np.float32)
                recent_exog_unscaled_df = exog_val_unscaled.iloc[-lookback_window:].reset_index(drop=True)

                start_infer = time.time()
                forecast = recursive_inference_dynamic_exog(
                    model=model,
                    target_scaler=scaler,
                    exog_scaler=exog_scaler,
                    exog_cols=EXOG_COLS,
                    history_target_unscaled=recent_target_unscaled,
                    history_exog_unscaled=recent_exog_unscaled_df,
                    future_exog_unscaled=exog_test_unscaled.reset_index(drop=True),
                    target_channel=0,
                    device=str(device),
                )
                infer_time = time.time() - start_infer
                
                # Metrics
                rmse = np.sqrt(mean_squared_error(test, forecast))
                mae = mean_absolute_error(test, forecast)
                bias = np.mean(forecast - test)
                
                # POCID
                diff_original = test[1:] - test[:-1]
                diff_pred = forecast[1:] - forecast[:-1]
                is_positive = (diff_original * diff_pred) > 0
                pocid = is_positive.sum() / len(is_positive) if len(is_positive) > 0 else 0.0
                
                # Score
                score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)

                strategy_forecasts[strategy] = forecast
                strategy_rmses[strategy] = rmse
                strategy_maes[strategy] = mae
                strategy_biases[strategy] = bias
                strategy_scores[strategy] = score
                strategy_pocids[strategy] = pocid
                
                results.append({
                    'seed': seed,
                    'strategy': strategy,
                    'item_id': item_id,
                    'store_id': store_id,
                    'rmse': rmse,
                    'mae': mae,
                    'bias': bias,
                    'composite_score': score,
                    'pocid': pocid,
                    'train_time': train_time,
                    'inference_time': infer_time,
                    'best_epoch': best_epoch
                })

            # --- Plot Forecast Comparisons ---
            if ENABLE_PLOTS:
                plot_dir = os.path.join(script_dir, f'grid_search_plots/seed_{seed}/{loss_type}')
                os.makedirs(plot_dir, exist_ok=True)

                #strategies_str = "_".join(strategies)
                strategies_str = "_"
                plot_path = os.path.join(plot_dir, f'item{item_id}_store{store_id}_{strategies_str}_comparison.html')

                plot_results(train, val, test, strategy_forecasts,
                             train_index, val_index, test_index,
                             strategy_train_losses, strategy_val_losses,
                             target_col=TARGET_COL, title=f'MLP Strategies Comparison - Item {item_id} Store {store_id} (Seed {seed})',
                             save_path=plot_path,
                             rmse=strategy_rmses, mae=strategy_maes, bias=strategy_biases,
                             score=strategy_scores, pocid=strategy_pocids, df_full=df_product)

            # Incremental save so a partial/killed run still leaves a CSV behind.
            pd.DataFrame(results).to_csv(seed_csv_path, index=False)

        # --- Persist per-seed results into MLP_Results/seed_{seed}/{loss_type}/results.csv ---
        pd.DataFrame(results).to_csv(seed_csv_path, index=False)
        per_seed_csv_paths.append(seed_csv_path)
        all_results.extend(results)
        print(f"[seed {seed}] saved per-seed results to {seed_csv_path}")

    # --- Merge all per-seed CSVs and pick the best model per (item, store) ---
    merged_df = pd.concat([pd.read_csv(p) for p in per_seed_csv_paths], ignore_index=True)
    merged_csv_path = os.path.join(results_root, 'mlp_strategies_results.csv')
    merged_df.to_csv(merged_csv_path, index=False)

    best_idx = merged_df.groupby(['item_id', 'store_id', 'strategy'])['composite_score'].idxmin()
    best_df = merged_df.loc[best_idx].reset_index(drop=True)
    best_csv_path = os.path.join(results_root, 'mlp_strategies_best_per_product.csv')
    best_df.to_csv(best_csv_path, index=False)

    print(f"Merged results: {merged_csv_path}")
    print(f"Best-per-product (by composite_score): {best_csv_path}")
    print("Experiments completed.")

if __name__ == "__main__":
    main()
