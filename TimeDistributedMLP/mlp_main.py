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
from train import TrainConfig, train_mlp_forecaster
from inference import recursive_inference_dynamic_exog
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join(script_dir, '..', 'dataset', 'data_andre.feather')
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
seeds = [42]
#seeds = [42, 1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]
#seeds = [42, 1000, 26008, 907969]  # Add more seeds as needed
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_type = 'MSELoss'

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
    
    target_products = [26002]
    products = df[df['item_id'].isin(target_products)][['item_id', 'store_id']].drop_duplicates().values[:5]
    results = []
    
    os.makedirs('best_models', exist_ok=True)
    os.makedirs('grid_search_plots', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    criterion2 = nn.MSELoss()

    for seed in seeds:
        for item_id, store_id in products:
            df_product = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].copy()
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

            # Keep UNSCALED copies — needed by the dynamic inference loop.
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

                # Write scaled values back so train.py sees scaled features.
                exog_col_idx = df_product.columns.get_indexer(EXOG_COLS)
                df_product.iloc[train_slice, exog_col_idx] = exog_train_scaled[EXOG_COLS].values
                df_product.iloc[val_slice, exog_col_idx] = exog_val_scaled[EXOG_COLS].values
                df_product.iloc[test_slice, exog_col_idx] = exog_test_scaled[EXOG_COLS].values
            else:
                exog_scaler = None
            
            input_size = 1 + len(EXOG_COLS)
            
            # Dataloaders and Dataset extraction deleted since the updated `train.py` functions use df & TrainConfig directly.

            train_index = df_product[DATE_COL][train_slice].values
            val_index = df_product[DATE_COL][val_slice].values
            test_index = df_product[DATE_COL][test_slice].values

            # Removed manual MLP instantiation here because it's now created inside the train script

            model_dir = f'best_models/seed_{seed}/{loss_type}'
            os.makedirs(model_dir, exist_ok=True)
            model_path = f'{model_dir}/mlp_item{item_id}_{store_id}.pth'

            # Prepare global config for this strategy
            cfg = TrainConfig(
                lookback=lookback_window,
                horizon=1,
                batch_size=BATCH_SIZE,
                train_size=train_size,
                val_size=val_size,
                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                epochs=EPOCHS,
                dropout=DROPOUT,
                patience=PATIENCE,
                hidden_sizes=HIDDEN_SIZES,
                device=str(DEVICE)
            )

            start_train = time.time()
            model, _, t_losses, v_losses, best_epoch = train_mlp_forecaster(
                df=df_product, cfg=cfg, seed=seed, loss_type=LOSS_TYPE,
                product_id=f"{item_id}_{store_id}", scaler=scaler, target_channel=0,
                target_col=TARGET_COL, exog_cols=EXOG_COLS, test_size=forecast_horizon
            )
            train_time = time.time() - start_train

            # Load best model
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))

            # Leak-safe recursive inference: lag/rolling cols are recomputed
            # from the running prediction buffer, never from ground-truth test.
            # Full train+val target passed so lag_364 resolves correctly.
            recent_target_unscaled = np.concatenate([train, val]).astype(np.float32)
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
            results.append({
                'seed': seed,
                'item_id': item_id,
                'store_id': store_id,
                'rmse': rmse,
                'mae': mae,
                'train_time': train_time,
                'inference_time': infer_time,
                'best_epoch': best_epoch
            })

            # --- Plot Forecast Comparisons ---
            plot_dir = os.path.join(script_dir, f'grid_search_plots/seed_{seed}/{loss_type}')
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_path = os.path.join(plot_dir, f'item{item_id}_store{store_id}_comparison.html')
            
            plot_results(train, val, test, {'MLP': forecast}, 
                         train_index, val_index, test_index,
                         {'MLP': t_losses}, {'MLP': v_losses},
                         target_col=TARGET_COL, title=f'MLP Forecast - Item {item_id} Store {store_id} (Seed {seed})',
                         save_path=plot_path,
                         rmse={'MLP': rmse}, mae={'MLP': mae}, bias={'MLP': bias},
                         score={'MLP': score}, pocid={'MLP': pocid}, df_full=df_product)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(script_dir, 'mlp_results.csv'), index=False)
    print("Experiments completed. Results saved to mlp_results.csv.")

if __name__ == "__main__":
    main()
