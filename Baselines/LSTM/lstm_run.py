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
from model_utils.plots import plot_results
from utils import generate_exogenous_features
from lstm import LSTM
from dataset import TimeSeriesDataset
from LSTM.lstm_train import train_model
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
lookback_window = 30     # >= max lag (lag_30): the recursive rollout rebuilds lags from this window

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
#seeds = [57]
seeds = [42,1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]  # Add more seeds as needed
 
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
            
            exog_scaler = MinMaxScaler()
            exog_train_scaled = exog_scaler.fit_transform(exog_train)
            exog_val_scaled = exog_scaler.transform(exog_val)
            exog_test_scaled = exog_scaler.transform(exog_test)
            
            input_size = 1 + len(EXOG_COLS)
            
            train_dataset = TimeSeriesDataset(train_scaled, exog_train_scaled, lookback_window)
            val_dataset = TimeSeriesDataset(val_scaled, exog_val_scaled, lookback_window)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            
            train_index = df_product[DATE_COL][train_slice].values
            val_index = df_product[DATE_COL][val_slice].values
            test_index = df_product[DATE_COL][test_slice].values

            print(f"\n--- LSTM | Seed: {seed}, Item: {item_id}, Store: {store_id} ---")
            model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=150//3)

            model_dir = f'best_models/seed_{seed}/{loss_type}'
            os.makedirs(model_dir, exist_ok=True)
            model_path = f'{model_dir}/lstm_item{item_id}_store{store_id}.pth'

            # Train with early stopping on the validation loss; best-validation weights are saved.
            model, t_losses, v_losses, best_epoch, train_time = train_model(
                EPOCHS, model, train_loader, val_loader, EXOG_COLS,
                criterion, criterion2, optimizer, device, model_path, scheduler, 150)

            # Load best model
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))

            # Inference
            forecast, infer_time = recursive_inference(
                model, test_start_idx, lookback_window, val_scaled, exog_val_scaled, exog_test_scaled, exog_test,
                scaler, exog_scaler, df_product, device, EXOG_COLS, forecast_horizon, item_id, store_id, loss_type, script_dir
            )

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

            # --- Plot Forecast ---
            plot_dir = os.path.join(script_dir, f'grid_search_plots/seed_{seed}/{loss_type}')
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f'item{item_id}_store{store_id}.html')

            plot_results(train, val, test, {'LSTM': forecast},
                         train_index, val_index, test_index,
                         {'LSTM': t_losses}, {'LSTM': v_losses},
                         target_col=TARGET_COL, title=f'LSTM - Item {item_id} Store {store_id} (Seed {seed})',
                         save_path=plot_path,
                         rmse={'LSTM': rmse}, mae={'LSTM': mae}, bias={'LSTM': bias},
                         score={'LSTM': score}, pocid={'LSTM': pocid}, df_full=df_product)

    results_df = pd.DataFrame(results)
    results_df.to_csv('lstm_results.csv', index=False)
    print("Experiments completed. Results saved to lstm_results.csv.")

if __name__ == "__main__":
    main()
