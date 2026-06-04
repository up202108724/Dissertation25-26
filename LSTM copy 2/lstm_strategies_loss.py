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

sys.path.append(os.path.abspath('..'))
from plots import plot_results
from utils import generate_exogenous_features
from lstm import LSTM
from dataset import TimeSeriesDataset
from LSTM.lstm_train import train_model, train_model_best_train_loss, train_model_combined, train_model_expanding_window, train_model_sliding_window
#from train import train_model_selected_epochs
from LSTM.lstm_inference import recursive_inference
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, '..', 'dataset', 'data_andre.feather')
DATE_COL = 'date'
TARGET_COL = 'value'

train_size = 455
val_size = 154
forecast_horizon = 152
lookback_window = 7

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

batch_size = 32
hidden_size = 32
num_layers = 1
dropout = 0.0
EPOCHS = 1000
LEARNING_RATE = 0.001
seeds = [42]
#seeds = [42,1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]  # Add more seeds as needed

# -----------------------------------------------------------------------------
# Loss Registry
# -----------------------------------------------------------------------------
LOSS_CONFIGS = {
    'MSELoss':   lambda: nn.MSELoss(),
    'L1Loss':    lambda: nn.L1Loss(),
    'HuberLoss': lambda: nn.HuberLoss(delta=1.0),
}

STRATEGY = 'best_val'  # Fixed strategy for this grid

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
            
            exog_scaler = MinMaxScaler()
            exog_train_scaled = exog_scaler.fit_transform(exog_train)
            exog_val_scaled = exog_scaler.transform(exog_val)
            exog_test_scaled = exog_scaler.transform(exog_test)
            
            input_size = 1 + len(EXOG_COLS)
            
            train_dataset = TimeSeriesDataset(train_scaled, exog_train_scaled, lookback_window)
            val_dataset = TimeSeriesDataset(val_scaled, exog_val_scaled, lookback_window)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            
            combined_train_val = np.concatenate([train_scaled, val_scaled])
            combined_exog = np.concatenate([exog_train_scaled, exog_val_scaled]) if exog_train_scaled is not None else None
            combined_dataset = TimeSeriesDataset(combined_train_val, combined_exog, lookback_window)
            combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)

            loss_forecasts = {}
            loss_train_losses = {}
            loss_val_losses = {}
            loss_rmses = {}
            loss_maes = {}
            loss_biases = {}
            loss_scores = {}
            loss_pocids = {}

            train_index = df_product[DATE_COL][train_slice].values
            val_index = df_product[DATE_COL][val_slice].values
            test_index = df_product[DATE_COL][test_slice].values

            for loss_type, loss_fn in LOSS_CONFIGS.items():
                print(f"\n--- Loss: {loss_type}, Seed: {seed}, Item: {item_id}, Store: {store_id} ---")
                criterion = loss_fn()
                criterion2 = nn.MSELoss()  # Always use MSE for val monitoring so losses are comparable

                model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=150//3)
                
                model_dir = f'best_models/seed_{seed}/{loss_type}/{STRATEGY}'
                os.makedirs(model_dir, exist_ok=True)
                model_path = f'{model_dir}/lstm_item{item_id}_store{store_id}.pth'
                
                model, t_losses, v_losses, best_epoch, train_time = train_model(
                    seed, EPOCHS, model, train_loader, val_loader, EXOG_COLS,
                    criterion, criterion2, optimizer, device, model_path, scheduler, 150)

                loss_train_losses[loss_type] = t_losses
                loss_val_losses[loss_type] = v_losses

                # Load best model
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path))
                
                # Inference
                forecast, infer_time = recursive_inference(
                    model, test_start_idx, lookback_window, val_scaled, exog_val_scaled, exog_test_scaled, exog_test,
                    scaler, exog_scaler, df_product, device, EXOG_COLS, forecast_horizon, seed, STRATEGY, item_id, store_id, loss_type, script_dir
                )
                
                # Metrics
                rmse = np.sqrt(mean_squared_error(test, forecast))
                mae = mean_absolute_error(test, forecast)
                bias = np.mean(forecast - test)
                
                diff_original = test[1:] - test[:-1]
                diff_pred = forecast[1:] - forecast[:-1]
                is_positive = (diff_original * diff_pred) > 0
                pocid = is_positive.sum() / len(is_positive) if len(is_positive) > 0 else 0.0
                
                score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)

                loss_forecasts[loss_type] = forecast
                loss_rmses[loss_type] = rmse
                loss_maes[loss_type] = mae
                loss_biases[loss_type] = bias
                loss_scores[loss_type] = score
                loss_pocids[loss_type] = pocid
                
                results.append({
                    'seed': seed,
                    'loss_type': loss_type,
                    'strategy': STRATEGY,
                    'item_id': item_id,
                    'store_id': store_id,
                    'rmse': rmse,
                    'mae': mae,
                    'train_time': train_time,
                    'inference_time': infer_time,
                    'best_epoch': best_epoch
                })

            # --- Plot all losses together for this item/seed ---
            plot_dir = os.path.join(script_dir, f'grid_search_plots/seed_{seed}/{STRATEGY}')
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_path = os.path.join(plot_dir, f'item{item_id}_store{store_id}_loss_comparison.html')
            
            plot_results(train, val, test, loss_forecasts,
                         train_index, val_index, test_index,
                         loss_train_losses, loss_val_losses,
                         target_col=TARGET_COL, title=f'LSTM Loss Comparison - Item {item_id} Store {store_id} (Seed {seed})',
                         save_path=plot_path,
                         rmse=loss_rmses, mae=loss_maes, bias=loss_biases,
                         score=loss_scores, pocid=loss_pocids, df_full=df_product)

    results_df = pd.DataFrame(results)
    results_df.to_csv('lstm_loss_comparison_results.csv', index=False)
    print("Experiments completed. Results saved to lstm_loss_comparison_results.csv.")

if __name__ == "__main__":
    main()
