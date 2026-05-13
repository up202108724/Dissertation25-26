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
from train import train_model, train_model_best_train_loss, train_model_combined, train_model_expanding_window, train_model_sliding_window, train_model_selected_epochs
from inference import recursive_inference
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
seeds = [57]
#seeds = [42,1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]  # Add more seeds as needed
 
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
    #strategies= ['selected_epochs']
    strategies = ['best_val', 'best_train_early_val', 'combined', 'expanding_window', 'sliding_window']
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
            
            combined_train_val = np.concatenate([train_scaled, val_scaled])
            combined_exog = np.concatenate([exog_train_scaled, exog_val_scaled]) if exog_train_scaled is not None else None
            combined_dataset = TimeSeriesDataset(combined_train_val, combined_exog, lookback_window)
            combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)

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
                model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=150//3)
                
                model_dir = f'best_models/seed_{seed}/{loss_type}/{strategy}'
                os.makedirs(model_dir, exist_ok=True)
                model_path = f'{model_dir}/lstm_item{item_id}_store{store_id}.pth'
                
                if strategy == 'best_val':
                    model, t_losses, v_losses, best_epoch, train_time = train_model(
                        seed, EPOCHS, model, train_loader, val_loader, EXOG_COLS, 
                        criterion, criterion2, optimizer, device, model_path, scheduler, 150)
                elif strategy == 'best_train_early_val':
                    model, t_losses, v_losses, best_epoch, train_time = train_model_best_train_loss(
                        seed, EPOCHS, model, train_loader, val_loader, EXOG_COLS, 
                        criterion, criterion2, optimizer, device, model_path, scheduler, 150)
                elif strategy == 'combined':
                    # First run standard train to find optimal epochs and best model via validation loss
                    model, t_init, v_losses, optimal_epoch, _ = train_model(
                        seed, EPOCHS, model, train_loader, val_loader, EXOG_COLS, 
                        criterion, criterion2, optimizer, device, model_path + "_temp.pth", scheduler, 150)
                    
                    # Load the best model weights
                    if os.path.exists(model_path + "_temp.pth"):
                        model.load_state_dict(torch.load(model_path + "_temp.pth"))
                    
                    # Retrain the same model on combined data for the same number of epochs
                    model, t_losses, train_time = train_model_combined(
                        seed, optimal_epoch, model, combined_loader, criterion, optimizer, device, model_path)
                    best_epoch = optimal_epoch
                elif strategy == 'selected_epochs':
                    # Ask or define a selected number of epochs, e.g., 200 (could be dynamically assigned)
                    selected_num_epochs = 200 
                    model, t_losses, best_train_loss, best_model_epoch, train_time = train_model_selected_epochs(
                        seed, selected_num_epochs, model, combined_loader, criterion, optimizer, device, model_path)
                    best_epoch = best_model_epoch
                    print(f"Best Train Loss: {best_train_loss}, at epoch: {best_model_epoch}")
                    v_losses = [] # No validation loss since it's train+val combined
                elif strategy == 'expanding_window':
                    model, t_losses, v_losses, best_epoch, train_time = train_model_expanding_window(
                        seed=seed, epochs=EPOCHS, model=model, 
                        full_train_scaled=combined_train_val, exog_scaled=combined_exog, 
                        seq_length=lookback_window, initial_train_size=train_size, 
                        val_step_size=30, batch_size=batch_size, 
                        criterion=criterion, criterion2=criterion2, 
                        optimizer=optimizer, device=device, final_model_path=model_path, 
                        dataset_class=TimeSeriesDataset, scheduler=scheduler, patience=150
                    )
                elif strategy == 'sliding_window':
                    model, t_losses, v_losses, best_epoch, train_time = train_model_sliding_window(
                        seed=seed, epochs=EPOCHS, model=model, 
                        full_train_scaled=combined_train_val, exog_scaled=combined_exog, 
                        seq_length=lookback_window, initial_train_size=train_size, 
                        val_step_size=30, batch_size=batch_size, 
                        criterion=criterion, criterion2=criterion2, 
                        optimizer=optimizer, device=device, final_model_path=model_path, 
                        dataset_class=TimeSeriesDataset, scheduler=scheduler, patience=150
                    )

                strategy_train_losses[strategy] = t_losses
                strategy_val_losses[strategy] = v_losses

                # Load best model
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path))
                
                # Inference
                forecast, infer_time = recursive_inference(
                    model, test_start_idx, lookback_window, val_scaled, exog_val_scaled, exog_test_scaled, exog_test,
                    scaler, exog_scaler, df_product, device, EXOG_COLS, forecast_horizon, seed, strategy, item_id, store_id, loss_type, script_dir
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
                    'train_time': train_time,
                    'inference_time': infer_time,
                    'best_epoch': best_epoch
                })

            # --- Plot Forecast Comparisons ---
            plot_dir = os.path.join(script_dir, f'grid_search_plots/seed_{seed}/{loss_type}')
            os.makedirs(plot_dir, exist_ok=True)
            
            #strategies_str = "_".join(strategies)
            strategies_str = "_"
            plot_path = os.path.join(plot_dir, f'item{item_id}_store{store_id}_{strategies_str}_comparison.html')
            
            plot_results(train, val, test, strategy_forecasts, 
                         train_index, val_index, test_index,
                         strategy_train_losses, strategy_val_losses,
                         target_col=TARGET_COL, title=f'LSTM Strategies Comparison - Item {item_id} Store {store_id} (Seed {seed})',
                         save_path=plot_path,
                         rmse=strategy_rmses, mae=strategy_maes, bias=strategy_biases,
                         score=strategy_scores, pocid=strategy_pocids, df_full=df_product)

    results_df = pd.DataFrame(results)
    results_df.to_csv('lstm_strategies_results.csv', index=False)
    print("Experiments completed. Results saved to lstm_strategies_results.csv.")

if __name__ == "__main__":
    main()
