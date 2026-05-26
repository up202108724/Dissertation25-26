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
from utils import generate_exogenous_features
from train import TrainConfig, train_mlp_forecaster
from inference import recursive_inference
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

EXOG_COLS = [
    # Cyclical Calendar Features 
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "dom_sin", "dom_cos", "wom_sin", "wom_cos", "month_sin", "month_cos", "quarter_sin", "quarter_cos", "woy_sin", "woy_cos",
 
    # Structural boundaries 
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
   
    # Trend Hint 
    "rolling_mean_7",
   
 
    # Holidays & Events (Crucial)
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_bridge_day"
]
'''
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
'''
batch_size = 32
hidden_sizes = (64, 32)
dropout = 0.2
EPOCHS = 1000
LEARNING_RATE = 0.001
#seeds = [42]
seeds = [42, 1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]  # Add more seeds as needed

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
    
    target_products = [26008, 907969, 907967, 213626]
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

            exog_train = df_product[EXOG_COLS][train_slice].values
            exog_val = df_product[EXOG_COLS][val_slice].values
            exog_test = df_product[EXOG_COLS][test_slice].values
            
            if len(EXOG_COLS) > 0:
                exog_scaler = MinMaxScaler()
                exog_train_scaled = exog_scaler.fit_transform(exog_train)
                exog_val_scaled = exog_scaler.transform(exog_val)
                exog_test_scaled = exog_scaler.transform(exog_test)
                
                # Apply to df_product so truth/target remains unscaled but exogs are scaled for train.py
                exog_indices = df_product.columns.get_indexer(EXOG_COLS)
                df_product.iloc[train_slice, exog_indices] = exog_train_scaled
                df_product.iloc[val_slice, exog_indices] = exog_val_scaled
                df_product.iloc[test_slice, exog_indices] = exog_test_scaled
            else:
                exog_scaler = None
                exog_train_scaled = exog_train
                exog_val_scaled = exog_val
                exog_test_scaled = exog_test
            
            input_size = 1 + len(EXOG_COLS)
            
            # Dataloaders and Dataset extraction deleted since the updated `train.py` functions use df & TrainConfig directly.

            train_index = df_product[DATE_COL][train_slice].values
            val_index = df_product[DATE_COL][val_slice].values
            test_index = df_product[DATE_COL][test_slice].values

            # Removed manual MLP instantiation here because it's now created inside the train script

            model_dir = f'best_models/seed_{seed}/{loss_type}'
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
                    device=str(device)
                )

            model, _, t_losses, v_losses, best_epoch = train_mlp_forecaster(
                        df=df_product, cfg=cfg, seed=seed, loss_type='mse', 
                        product_id=f"{item_id}_{store_id}", scaler=scaler, target_channel=0, val_ratio=0.2, 
                        hidden_sizes=hidden_sizes, target_col=TARGET_COL, exog_cols=EXOG_COLS, test_size=forecast_horizon)
            train_time = 0.0 
            # Load best model
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                
                # Inference
            recent_target = val[-lookback_window:].reshape(-1, 1)
            recent_exog_scaled = exog_val_scaled[-lookback_window:]
            recent_history = np.column_stack([recent_target, recent_exog_scaled])
                
            start_infer = time.time()
            forecast = recursive_inference(
                model=model,
                scaler=scaler,
                recent_history=recent_history,
                future_exog=exog_test_scaled,
                target_channel=0,
                device=str(device)
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
