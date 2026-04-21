import os
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
from graph2vecdataset import TimeSeriesDataset
from model_utils.utils import generate_exogenous_features, compute_metrics 
from model_utils.plots import plot_results #ensure available
from GNN.DynamicSimilarities.Graph2vec.generate_graph2vec import load_or_generate_embeddings
from train import train_model
from graph2vecinference import graph2vec_inference

# Constants
# Resolve DATA_PATH perfectly from the file directory upwards
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/data_andre.feather'))
DATE_COL = 'date'
TARGET_COL = 'value'

# Add the products and stores you want to iterate over
PRODUCTS_TO_TEST = [
    (907969, 6269),
    (26008, 6269),
    (907967, 6270)
]

# EXOG_COLS definition
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

def main():
    # Load and Preprocess Data (Once for all products)
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_feather(DATA_PATH)

    if DATE_COL in df.index.names:
        if DATE_COL in df.columns:
            df = df.reset_index(drop=True)
        else:
            df = df.reset_index()
    if df.index.name == DATE_COL:
        df = df.reset_index(drop=True)
    df = df.reset_index(drop=True)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, 'item_id', 'store_id']).reset_index(drop=True)

    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS)
    full_df = df.copy()

    for product_id, store_id in PRODUCTS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"PROCESSING PRODUCT {product_id} FOR STORE {store_id}")
        print(f"{'='*80}\n")
        
        # Filter for the specific product and store
        df = full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)].sort_values(DATE_COL).reset_index(drop=True)

        forecast_horizon = 152
        seq_length = 30
        train_size = 455
        val_size = 154
        lookback_window = 7 
        BATCH_SIZE = 32

        test_start_idx = len(df) - forecast_horizon
        val_start_idx = test_start_idx - val_size
        train_start_idx = val_start_idx - train_size

        train_slice = slice(train_start_idx, val_start_idx)
        val_slice = slice(val_start_idx, test_start_idx)
        test_slice = slice(test_start_idx, None)

        # Extract Target
        train = df[TARGET_COL][train_slice].values
        val = df[TARGET_COL][val_slice].values
        test = df[TARGET_COL][test_slice].values

        # Scale Target
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
        val_scaled = scaler.transform(val.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test.reshape(-1, 1)).flatten()

        # Extract Exogenous Variables
        if EXOG_COLS and len(EXOG_COLS) > 0:
            exog_train = df[EXOG_COLS][train_slice].values
            exog_val = df[EXOG_COLS][val_slice].values
            exog_test = df[EXOG_COLS][test_slice].values
            # Scale Exogenous Variables
            exog_scaler = MinMaxScaler()
            exog_train_scaled = exog_scaler.fit_transform(exog_train)
            exog_val_scaled = exog_scaler.transform(exog_val)
            exog_test_scaled = exog_scaler.transform(exog_test)
        else:
            exog_train_scaled = None
            exog_val_scaled = None
            exog_test_scaled = None
            exog_scaler = None

        # Grid Search Parameters Setup
        metrics = ['cid', 'spearman']
        percentiles = [0.5,1,2]
        window_sizes = [15]     
        step_sizes = [1]        
        enable_edges_opts = [True, False]

        USE_EMBEDDINGS = True
        USE_RESIDUALS = False
        MODEL_TYPE = 'ridge'
        seed = 2024
        EPOCHS = 1000
        PATIENCE = 100
        LEARNING_RATE = 0.001
        HIDDEN_SIZE = 32
        NUM_LAYERS = 1
        DROPOUT = 0.0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs('grid_search_plots', exist_ok=True)
        os.makedirs('best_models', exist_ok=True)

        # Grid Search Loop for current product
        for metric, percentile, window_size, step_size, enable_edges in itertools.product(
            metrics, percentiles, window_sizes, step_sizes, enable_edges_opts
        ):
            print(f"\n{'='*60}")
            print(f"Running Experiment: metric={metric}, percentile={percentile}, window_size={window_size}, enable_edges={enable_edges}")
            print(f"{'='*60}")

            if USE_EMBEDDINGS:
                graph_embeddings, graph2vec_model, csv_path = load_or_generate_embeddings(
                    product_id=product_id,
                    metric=metric,
                    window_size=window_size,
                    step_size=step_size,
                    threshold=None,
                    enable_edges_within_star=enable_edges,
                    percentile=percentile,
                    use_residuals=USE_RESIDUALS,
                    model_type=MODEL_TYPE
                )
                embedding_dim = graph_embeddings.shape[1] if len(graph_embeddings.shape) > 1 else 1

                padding = np.zeros((window_size - 1, embedding_dim))
                aligned_embeddings = np.vstack([padding, graph_embeddings])

                emb_train = aligned_embeddings[train_slice]
                emb_val = aligned_embeddings[val_slice]
            else:
                graph_embeddings = None
                graph2vec_model = None
                aligned_embeddings = None
                emb_train = None
                emb_val = None
                embedding_dim = 0
                
            input_size = 1 + (len(EXOG_COLS) if EXOG_COLS else 0) + embedding_dim

            # Combine features
            if EXOG_COLS and len(EXOG_COLS) > 0:
                if USE_EMBEDDINGS:
                    exog_train_combined = np.hstack([exog_train_scaled, emb_train])
                    exog_val_combined = np.hstack([exog_val_scaled, emb_val])
                else:
                    exog_train_combined = exog_train_scaled
                    exog_val_combined = exog_val_scaled
            else:
                if USE_EMBEDDINGS:
                    exog_train_combined = emb_train
                    exog_val_combined = emb_val
                else:
                    exog_train_combined = None
                    exog_val_combined = None

            train_dataset = TimeSeriesDataset(train_scaled, exog_train_combined, seq_length)
            use_pin_memory = torch.cuda.is_available()
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=use_pin_memory)
                
            val_dataset = TimeSeriesDataset(val_scaled, exog_val_combined, seq_length)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=use_pin_memory)

            # Initialize Model
            model = LSTM(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
            criterion = nn.MSELoss()
            criterion2 = nn.MSELoss()  
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE//3)

            if USE_EMBEDDINGS:
                prefix_star = "" if enable_edges else "star_"
                prefix = f"best_lstm_{prefix_star}{product_id}_{metric}_res_{MODEL_TYPE}" if USE_RESIDUALS else f"best_lstm_{prefix_star}{product_id}_{metric}"
                best_model_path = f'best_models/{prefix}_{window_size}_{step_size}_percentile_{percentile}.pth'      
                history_path = f'best_models/{prefix}_{window_size}_{step_size}_percentile_{percentile}_history.pkl'
            else:
                best_model_path = f'best_models/best_lstm_{product_id}_no_emb.pth'
                history_path = f'best_models/best_lstm_{product_id}_no_emb_history.pkl'

            if os.path.exists(best_model_path) and os.path.exists(history_path):
                print(f"Loading existing model from {best_model_path}...")
                model.load_state_dict(torch.load(best_model_path))
                with open(history_path, 'rb') as f:
                    history = pickle.load(f)
                    train_losses = history['train_losses']
                    val_losses = history['val_losses']
            else:
                print("Training new model...")
                model, train_losses, val_losses, best_epoch, train_time = train_model(
                    seed=seed, epochs=EPOCHS, model=model, 
                    train_loader=train_loader, val_loader=val_loader, 
                    exog_cols=EXOG_COLS, criterion=criterion, criterion2=criterion2, 
                    optimizer=optimizer, device=device, 
                    best_model_path=best_model_path, scheduler=scheduler, patience=PATIENCE
                )
                with open(history_path, 'wb') as f:
                    pickle.dump({
                        'train_losses': train_losses, 'val_losses': val_losses,
                        'best_epoch': best_epoch, 'train_time': train_time
                    }, f)

            # Explicitly load the best saved model before inference to guarantee clean pipeline state
            print(f"Loading best weights from {best_model_path} for inference...")
            model.load_state_dict(torch.load(best_model_path))

            exog_test_data = df[EXOG_COLS][test_slice].values

            print("Running Inference...")
            forecast, inference_time = graph2vec_inference(
                metric=metric, window_size=window_size, step_size=step_size,
                threshold=None, percentile=percentile, model=model,
                df=df, df_wide=None, cat_labels=None, date_col=DATE_COL,
                scaler=scaler, exog_scaler=exog_scaler,
                test_start_idx=test_start_idx, seq_length=seq_length,
                forecast_window=forecast_horizon, device=device,
                item_id=product_id, store_id=store_id, seed=seed,
                criterion="MSELoss", val_scaled=val_scaled, test_scaled=test_scaled,
                exog_val_scaled=exog_val_scaled, exog_test_scaled=exog_test_scaled,
                exog_test_raw=exog_test_data, exog_cols=EXOG_COLS,
                save_plot_path=None,
                node_embeddings=aligned_embeddings if USE_EMBEDDINGS else None,
                graph2vec_model=graph2vec_model if USE_EMBEDDINGS else None,
                enable_edges_within_star=enable_edges
            )

            if USE_EMBEDDINGS:
                save_plot_path = f"grid_search_plots/item_{product_id}_store_{store_id}_{prefix}_window_{window_size}_step_{step_size}_percentile_{percentile}.png"
            else:
                save_plot_path = f"grid_search_plots/item_{product_id}_store_{store_id}_no_emb.png"

            valid_mask = ~np.isnan(forecast)
            valid_test = test[valid_mask]
            valid_forecast = np.array(forecast)[valid_mask]

            rmse, mae, bias, score, pocid = None, None, None, None, None
            try:
                rmse, mae, bias, score, pocid = compute_metrics(valid_test, valid_forecast)
            except Exception as e:
                if len(valid_test) > 0:
                    rmse = np.sqrt(mean_squared_error(valid_test, valid_forecast))
                    mae = mean_absolute_error(valid_test, valid_forecast)
                    bias = np.mean(valid_forecast - valid_test)
                    score = r2_score(valid_test, valid_forecast)

            train_index = df[DATE_COL][train_slice].values
            val_index = df[DATE_COL][val_slice].values
            test_index = df[DATE_COL][test_slice].values
                
            emb_title = f'Graph2Vec ({metric} | pct: {percentile} | star: {enable_edges})'
            plot_results(train, val, test, forecast, train_index, val_index, test_index,
                         train_losses, val_losses, metric=metric, embedding_strategy='graph2vec',
                         window_size=window_size, step_size=step_size, threshold=None, percentile=percentile,
                         enable_edges_within_star=enable_edges, target_col=TARGET_COL, 
                         title=f'LSTM Forecast {emb_title} (Item={product_id})',
                         save_path=save_plot_path, rmse=rmse, mae=mae, bias=bias, score=score, pocid=pocid)

            print(f"Finished {metric} @ {percentile} -> RMSE: {rmse:.4f}\n")

if __name__ == '__main__':
    main()
