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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths & Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lstm import LSTM
from dataset import TimeSeriesDataset
from model_utils.utils import generate_exogenous_features, compute_metrics 
from plots import plot_results
from train import train_model

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/data_andre.feather'))
DATE_COL = 'date'
TARGET_COL = 'value'
SEEDS = [42]

PRODUCTS_TO_TEST = [
  (26008, 6269),
  (907969, 6269),
  (907967, 6269),
  (213626, 6269),
]

EXOG_COLS = [
    "day_of_week", "day_of_month", "week_of_year", "week_of_month",
    "month", "quarter", "is_weekend",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_monday", "is_friday",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_pre_holiday_1", "is_pre_holiday_2", "is_pre_holiday_3", "is_pre_holiday_7",
    "is_post_holiday_1", "is_post_holiday_2", "is_post_holiday_3", "is_post_holiday_7",
    "is_bridge_day",
]

def simple_autoregressive_inference(model, seq_length, forecast_window, device, 
                                  val_scaled, test_scaled, exog_val_scaled, exog_test_scaled, 
                                  exog_test_raw, exog_cols, scaler, exog_scaler, date_col, df, test_start_idx):
    model.eval()
    forecast = []
    
    current_seq = val_scaled[-seq_length:].tolist()
    if exog_cols and len(exog_cols) > 0:
        current_exog_seq = exog_val_scaled[-seq_length + 1:].tolist() + [exog_test_scaled[0].tolist()]
    else:
        current_exog_seq = []
        
    start_time = time.time()
    with torch.no_grad():
        for step in range(forecast_window):
            features_to_stack = [np.array(current_seq).reshape(-1, 1)]
            if exog_cols and len(exog_cols) > 0:
                features_to_stack.append(np.array(current_exog_seq))
                
            x_np = np.column_stack(features_to_stack)
            x_tensor = torch.FloatTensor(x_np).unsqueeze(0).to(device)
            
            pred = model(x_tensor).cpu().numpy()[0, 0]
            forecast.append(pred)
            current_seq.pop(0)
            current_seq.append(pred)
            
            if exog_cols and len(exog_cols) > 0:
                current_exog_seq.pop(0)
                if step + 1 < forecast_window:
                    current_exog_seq.append(exog_test_scaled[step + 1].tolist())
                else:
                    current_exog_seq.append(exog_test_scaled[-1].tolist())
                    
    inf_time = time.time() - start_time
    forecast_unscaled = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    return forecast_unscaled, inf_time


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_feather(DATA_PATH)

    if DATE_COL in df.index.names:
        if DATE_COL in df.columns:
            df = df.reset_index(drop=True)
        else:
            df = df.reset_index()
    if df.index.name == DATE_COL:
        df = df.reset_index(drop=True)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, 'item_id', 'store_id']).reset_index(drop=True)

    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS)
    full_df = df.copy()

    for product_id, store_id in PRODUCTS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"PROCESSING PRODUCT {product_id} FOR STORE {store_id}")
        print(f"{'='*80}\n")
        
        df = full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)].sort_values(DATE_COL).reset_index(drop=True)

        forecast_horizon = 152
        train_size = 455
        val_size = 154

        required_rows = forecast_horizon + val_size + train_size
        if len(df) < required_rows:
            print(f"Skipping Product {product_id}: Found {len(df)} rows, but {required_rows} required.")
            continue

        test_start_idx = len(df) - forecast_horizon
        val_start_idx = test_start_idx - val_size
        train_start_idx = val_start_idx - train_size

        train_slice = slice(train_start_idx, val_start_idx)
        val_slice = slice(val_start_idx, test_start_idx)
        test_slice = slice(test_start_idx, None)

        train = df[TARGET_COL][train_slice].values
        val = df[TARGET_COL][val_slice].values
        test = df[TARGET_COL][test_slice].values

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
        val_scaled = scaler.transform(val.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test.reshape(-1, 1)).flatten()

        if EXOG_COLS and len(EXOG_COLS) > 0:
            exog_train = df[EXOG_COLS][train_slice].values
            exog_val = df[EXOG_COLS][val_slice].values
            exog_test = df[EXOG_COLS][test_slice].values
            exog_scaler = MinMaxScaler()
            exog_train_scaled = exog_scaler.fit_transform(exog_train)
            exog_val_scaled = exog_scaler.transform(exog_val)
            exog_test_scaled = exog_scaler.transform(exog_test)
        else:
            exog_train_scaled = exog_val_scaled = exog_test_scaled = exog_scaler = None

        for seed in SEEDS:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            print(f"\n--- RUNNING WITH SEED {seed} ---\n")

            LEARNING_RATES = [0.001]
            HIDDEN_SIZES = [32, 64]
            NUM_LAYERS_OPTS = [1]
            DROPOUTS = [0.0]
            SEQ_LENGTHS = [30]
            BATCH_SIZES = [32]
            EPOCHS = 1000
            PATIENCE = 100

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            grid_search_plots_dir = os.path.join(script_dir, 'grid_search_plots', f'seed_{seed}', f'item_{product_id}')
            best_models_seed_dir = os.path.join(script_dir, 'best_models', f'seed_{seed}')
            
            os.makedirs(grid_search_plots_dir, exist_ok=True)
            os.makedirs(best_models_seed_dir, exist_ok=True)

            results_by_config = {
                'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {}
            }

            iterator = itertools.product(LEARNING_RATES, HIDDEN_SIZES, NUM_LAYERS_OPTS, DROPOUTS, SEQ_LENGTHS, BATCH_SIZES)
            
            for lr, hidden_size, num_layers, dropout, seq_len, batch_size in iterator:
                config_label = f"lr={lr}|h={hidden_size}|n={num_layers}|d={dropout}|seq={seq_len}"
                print(f"\n{'='*60}\nRunning Config: {config_label}\n{'='*60}")

                input_size = 1 + (len(EXOG_COLS) if EXOG_COLS else 0)

                train_dataset = TimeSeriesDataset(
                    target_data=train_scaled, 
                    exog_data=exog_train_scaled if EXOG_COLS and len(EXOG_COLS) > 0 else None, 
                    seq_length=seq_len
                )
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
            
                val_dataset = TimeSeriesDataset(
                    target_data=val_scaled, 
                    exog_data=exog_val_scaled if EXOG_COLS and len(EXOG_COLS) > 0 else None, 
                    seq_length=seq_len
                )
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

                model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
                criterion = nn.MSELoss()
                criterion2 = nn.MSELoss()  
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE//3)

                best_model_path = os.path.join(best_models_seed_dir, f'best_lstm_{product_id}_lr{lr}_h{hidden_size}_n{num_layers}_seq{seq_len}_seed_{seed}.pth')
                history_path = os.path.join(best_models_seed_dir, f'best_lstm_{product_id}_lr{lr}_h{hidden_size}_n{num_layers}_seq{seq_len}_seed_{seed}_history.pkl')

                if os.path.exists(best_model_path) and os.path.exists(history_path):
                    print(f"Loading existing model from {best_model_path}...")
                    model.load_state_dict(torch.load(best_model_path))
                    with open(history_path, 'rb') as f:
                        history = pickle.load(f)
                        train_losses, val_losses = history['train_losses'], history['val_losses']
                else:
                    print("Training new model...")
                    model, train_losses, val_losses, best_epoch, train_time = train_model(
                        seed=seed, epochs=EPOCHS, model=model, train_loader=train_loader, val_loader=val_loader, 
                        exog_cols=EXOG_COLS, criterion=criterion, criterion2=criterion2, optimizer=optimizer, 
                        device=device, best_model_path=best_model_path, scheduler=scheduler, patience=PATIENCE
                    )
                    with open(history_path, 'wb') as f:
                        pickle.dump({'train_losses': train_losses, 'val_losses': val_losses, 'best_epoch': best_epoch, 'train_time': train_time}, f)

                model.load_state_dict(torch.load(best_model_path))
                exog_test_data = df[EXOG_COLS][test_slice].values

                forecast, inference_time = simple_autoregressive_inference(
                    model=model, seq_length=seq_len, forecast_window=forecast_horizon, device=device,
                    val_scaled=val_scaled, test_scaled=test_scaled, exog_val_scaled=exog_val_scaled, exog_test_scaled=exog_test_scaled,
                    exog_test_raw=exog_test_data, exog_cols=EXOG_COLS, scaler=scaler, exog_scaler=exog_scaler, date_col=DATE_COL,
                    df=df, test_start_idx=test_start_idx
                )

                valid_mask = ~np.isnan(forecast)
                valid_test, valid_forecast = test[valid_mask], np.array(forecast)[valid_mask]

                try:
                    rmse, mae, bias, score, pocid = compute_metrics(valid_test, valid_forecast)
                except Exception:
                    rmse = np.sqrt(mean_squared_error(valid_test, valid_forecast)) if len(valid_test) > 0 else 0
                    mae = mean_absolute_error(valid_test, valid_forecast) if len(valid_test) > 0 else 0
                    bias = np.mean(valid_forecast - valid_test) if len(valid_test) > 0 else 0
                    score = r2_score(valid_test, valid_forecast) if len(valid_test) > 0 else 0
                    pocid = 0

                results_by_config['forecasts'][config_label] = forecast
                results_by_config['train_losses'][config_label] = train_losses
                results_by_config['val_losses'][config_label] = val_losses
                results_by_config['rmse'][config_label] = rmse
                results_by_config['mae'][config_label] = mae
                results_by_config['bias'][config_label] = bias
                results_by_config['score'][config_label] = score
                results_by_config['pocid'][config_label] = pocid
            
                print(f"Finished {config_label} -> RMSE: {rmse:.4f}\n")

            train_index, val_index, test_index = df[DATE_COL][train_slice].values, df[DATE_COL][val_slice].values, df[DATE_COL][test_slice].values

            save_plot_path = os.path.join(grid_search_plots_dir, f"item_{product_id}_store_{store_id}_lstm_grid_seed_{seed}.html")
            grid_title = f'LSTM Hyperparameter Search (Item={product_id} | Seed={seed})'
            
            plot_results(train, val, test, results_by_config['forecasts'], train_index, val_index, test_index,
                         results_by_config['train_losses'], results_by_config['val_losses'], metric=None, embedding_strategy=None,
                         window_size=None, step_size=None, threshold=None, percentile=None,
                         target_col=TARGET_COL, title=grid_title, seed=seed,
                         save_path=save_plot_path, rmse=results_by_config['rmse'], mae=results_by_config['mae'], 
                         bias=results_by_config['bias'], score=results_by_config['score'], pocid=results_by_config['pocid'])

if __name__ == '__main__':
    main()
