import os
import random
import re
import sys
import time
import pickle
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths & Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lstm import LSTM
from graph2vecdataset import TimeSeriesDataset
from model_utils.utils import generate_exogenous_features, compute_metrics 
from plots import plot_results #ensure available
from generate_graph2vecwithadaptativethreshold import load_or_generate_embeddings, infer_metric_type
from train import train_model
from graph2vecinference_adaptativethreshold import graph2vec_inference

# Constants
# Resolve DATA_PATH perfectly from the file directory upwards
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/data_andre.feather'))
DATE_COL = 'date'
TARGET_COL = 'value'
#SEEDS = [42, 1000, 26008, 213626, 907969, 5219788,13451285]  # Add more seeds as needed
SEEDS =[42]
 
# Add the products and stores you want to iterate over
PRODUCTS_TO_TEST = [
  (26008, 6269),
  (907969,6269),
  (210036, 6269),
  (907967,6269),
  (213626,6269),
]

# EXOG_COLS definition
#EXOG_COLS = []

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

'''
grid_configs = [
    # Similaridades (Já existentes)
    {'metric': 'spearman', 'thresholds': [round(t, 3) for t in np.arange(0.6, 0.85, 0.001)]},
    {'metric': 'pearson', 'thresholds': [round(t, 3) for t in np.arange(0.6, 0.85, 0.001)]},
    {'metric': 'kendall', 'thresholds': [round(t, 3) for t in np.arange(0.6, 0.85, 0.001)]},
    
    # Distâncias Robustas e Lock-step
    {'metric': 'cid', 'thresholds': [round(t, 2) for t in np.arange(2.0, 3.5, 0.01)]},
    {'metric': 'manhattan', 'thresholds': [round(t, 2) for t in np.arange(4.0, 10.0, 0.1)]},
    {'metric': 'lorentzian', 'thresholds': [round(t, 2) for t in np.arange(1.0, 5.0, 0.1)]},
    
    # Distâncias Elásticas (Elastic)
    {'metric': 'dtw', 'thresholds': [round(t, 2) for t in np.arange(1.5, 4.0, 0.05)]},
    {'metric': 'twed', 'thresholds': [round(t, 2) for t in np.arange(2.0, 8.0, 0.2)]},
    {'metric': 'erp', 'thresholds': [round(t, 2) for t in np.arange(2.0, 8.0, 0.2)]},
    
    # Baseadas em Forma e Deslizamento (Sliding)
    {'metric': 'sbd', 'thresholds': [round(t, 3) for t in np.arange(0.05, 0.5, 0.01)]},
    {'metric': 'stid', 'thresholds': [round(t, 2) for t in np.arange(1.5, 3.5, 0.05)]},
    
    # Baseadas em Atributos (Feature-based)
    {'metric': 'catch22', 'thresholds': [round(t, 1) for t in np.arange(2.0, 15.0, 0.5)]},
]
'''
grid_configs = [
    # Distâncias Robustas e Lock-step
    #{'metric': 'cid', 'percentiles': [0.5, 1, 2]},
    #{'metric': 'amplitude_offset', 'percentiles': [0.5, 1, 2]},
   
    {'metric': 'cid', 'thresholds': [round(t, 2) for t in np.arange(2.0, 3.2, 0.01)]},
    # Distâncias Robustas e Lock-step
    {'metric': 'amplitude_offset', 'thresholds': [round(t, 2) for t in np.arange(2.0, 3.5, 0.01)]},
]

window_sizes = [15]     
step_sizes = [1]
enable_edges_opts = [True]
enable_second_degree_opts = [False]  # We will keep this False for the main analysis, but you can set to True to include second-degree neighbors in the graph construction
USE_RESIDUALS = False
MODEL_TYPE = 'ridge'
EPOCHS = 1000
PATIENCE = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.0
SAVE_MODELS = False
SAVE_PLOTS = True
USE_EMBEDDINGS = True
SAVE_EMBEDDINGS = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    # Pre-generate df_wide and category labels for inference
    cat_labels_dict = full_df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict() if 'cat_label' in full_df.columns else {}
    df_wide_global = full_df.pivot_table(index='item_id', columns=DATE_COL, values=TARGET_COL, aggfunc='sum').fillna(0)
    df_wide_global.columns = pd.to_datetime(df_wide_global.columns).strftime('%Y-%m-%d')

    # Globally define train split based on identical logic for all items
    L = len(df_wide_global.columns)
    forecast_horizon_global = 152
    val_size_global = 154
    train_size_global = 455
    global_train_start_idx = L - forecast_horizon_global - val_size_global - train_size_global
    global_val_start_idx = L - forecast_horizon_global - val_size_global

    # Build dictionary of local StandardScalers and transform df_wide_global
    product_scalers = {}
    
    # We ensure we don't drop out of bounds if df is shorter than assumed
    if global_train_start_idx < 0: global_train_start_idx = 0
    train_df_wide = df_wide_global.iloc[:, global_train_start_idx:global_val_start_idx]
    
    df_wide_scaled = df_wide_global.copy()
    for item_id_iter in df_wide_global.index:
        z_scaler = StandardScaler()
        # fit on training window for this item
        train_ts = train_df_wide.loc[item_id_iter].values.reshape(-1, 1)
        z_scaler.fit(train_ts)
        
        product_scalers[item_id_iter] = z_scaler
        
        # Transform the entire continuous history for the graph
        full_ts = df_wide_global.loc[item_id_iter].values.reshape(-1, 1)
        df_wide_scaled.loc[item_id_iter] = z_scaler.transform(full_ts).flatten()

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

        required_rows = forecast_horizon + val_size + train_size
        if len(df) < required_rows:
            print(f"Skipping Product {product_id} at Store {store_id}: Found {len(df)} rows, but {required_rows} are required for the splits.")
            continue

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

        for seed in SEEDS:
            # Set all seeds here
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            print(f"\n--- RUNNING WITH SEED {seed} ---\n")
            
            # Get the directory where `graph2vec_lstm.py` is located to anchor our save paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            grid_search_plots_dir = os.path.join(script_dir, 'grid_search_plots', f'seed_{seed}')
            best_models_seed_dir = os.path.join(script_dir, 'best_models', f'seed_{seed}')
            
            os.makedirs(grid_search_plots_dir, exist_ok=True)
            os.makedirs(best_models_seed_dir, exist_ok=True)

            # 1. Run Baseline (no embeddings) first
            all_configs = [{'metric': 'no_emb'}] + grid_configs if USE_EMBEDDINGS else [{'metric': 'no_emb'}]
            base_forecast, base_train_losses, base_val_losses = None, None, None
            base_rmse, base_mae, base_bias, base_score, base_pocid = None, None, None, None, None

            for config in all_configs:
                metric = config['metric']
                thresholds = config.get('thresholds', [None])
                percentiles = config.get('percentiles', [None])
                
                results_by_w_s = {}

                if metric == 'no_emb':
                    is_threshold_mode = False
                    iterator = [(None, 15, 1, False, False)]
                else:
                    is_threshold_mode = thresholds is not None and thresholds != [None]
                    params = thresholds if is_threshold_mode else percentiles
                    iterator = itertools.product(params, window_sizes, step_sizes, enable_edges_opts, enable_second_degree_opts)
            
                for param_val, window_size, step_size, enable_edges, enable_second_degree in iterator:
                    use_embeddings = (metric != 'no_emb')
                    
                    current_threshold = param_val if use_embeddings and is_threshold_mode else None
                    current_percentile = param_val if use_embeddings and not is_threshold_mode else None

                    key = (param_val, window_size, step_size)
                    if key not in results_by_w_s:
                        results_by_w_s[key] = {
                            'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                            'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {},
                            'threshold': None
                        }
                        if metric != 'no_emb' and base_forecast is not None:
                            results_by_w_s[key]['forecasts']["No Embeddings"] = base_forecast
                            results_by_w_s[key]['train_losses']["No Embeddings"] = base_train_losses
                            results_by_w_s[key]['val_losses']["No Embeddings"] = base_val_losses
                            results_by_w_s[key]['rmse']["No Embeddings"] = base_rmse
                            results_by_w_s[key]['mae']["No Embeddings"] = base_mae
                            results_by_w_s[key]['bias']["No Embeddings"] = base_bias
                            results_by_w_s[key]['score']["No Embeddings"] = base_score
                            results_by_w_s[key]['pocid']["No Embeddings"] = base_pocid

                    print(f"\n{'='*60}")
                    if use_embeddings:
                        param_str = f"threshold={current_threshold}" if is_threshold_mode else f"percentile={current_percentile}"
                        print(f"Running Experiment: metric={metric}, {param_str}, window_size={window_size}, enable_edges={enable_edges}, 2nd_degree={enable_second_degree}")
                    else:
                        print("Running Experiment: BASELINE (no graph embeddings)")
                    print(f"{'='*60}")

                    fixed_threshold = None
                    if use_embeddings:
                        metric_type = infer_metric_type(metric)
                        
                        # Use norm-scaled df for distances
                        distance_metrics = ['euclidean','manhattan', 'hamming', 'amplitude_offset', 'slope_consistency', 'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
                        current_df_wide = df_wide_scaled if metric in distance_metrics else df_wide_global
                        
                        graph_embeddings, graph2vec_model, csv_path, build_time, emb_time, fixed_threshold = load_or_generate_embeddings(
                            product_id=product_id,
                            metric=metric,
                            metric_type=metric_type,
                            window_size=window_size,
                            step_size=step_size,
                            threshold=current_threshold if is_threshold_mode else None,
                            enable_edges_within_star=enable_edges,
                            enable_second_degree=enable_second_degree,
                            percentile=current_percentile if not is_threshold_mode else None,
                            use_residuals=USE_RESIDUALS,
                            model_type=MODEL_TYPE,
                            seed=seed,
                            train_end_idx=val_start_idx,
                            df=current_df_wide,
                            cat_labels=cat_labels_dict,
                            save_embeddings=SAVE_EMBEDDINGS
                        )
                        print(f"Resolved graph threshold={current_threshold}: {fixed_threshold}")
                        print(f"Embedding file: {csv_path}")
                        results_by_w_s[key]['threshold'] = fixed_threshold
                        
                            
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

                    train_dataset = TimeSeriesDataset(
                        target_data=train_scaled, 
                        exog_data=exog_train_scaled if EXOG_COLS and len(EXOG_COLS) > 0 else None, 
                        seq_length=seq_length,
                        embeddings=emb_train,
                        graph_window_size=window_size
                    )
                    use_pin_memory = torch.cuda.is_available()
                    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=use_pin_memory)
                
                    val_dataset = TimeSeriesDataset(
                        target_data=val_scaled, 
                        exog_data=exog_val_scaled if EXOG_COLS and len(EXOG_COLS) > 0 else None, 
                        seq_length=seq_length,
                        embeddings=emb_val,
                        graph_window_size=window_size
                    )
                    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=use_pin_memory)

                    # Initialize Model with fixed seed for determinism across iterations!
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)
                        
                    model = LSTM(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
                    criterion = nn.MSELoss()
                    criterion2 = nn.MSELoss()  
                    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE//3)

                    if not use_embeddings:
                        model_dir_label = "no_emb"
                    else:
                        model_dir_label = f"th{current_threshold}" if is_threshold_mode else f"pct{current_percentile}"

                    best_models_dir = os.path.join(best_models_seed_dir, str(window_size), str(step_size), metric, model_dir_label)
                    os.makedirs(best_models_dir, exist_ok=True)

                    if use_embeddings:
                        if csv_path:
                            # Base the LSTM model name strictly on the embeddings filename
                            csv_basename = os.path.basename(csv_path)
                            base_name = csv_basename.replace('embeddings_', f'best_lstm_{product_id}_')
                            
                            if USE_RESIDUALS:
                                base_name = base_name.replace('.csv', f'_res_{MODEL_TYPE}.pth')
                            else:
                                base_name = base_name.replace('.csv', '.pth')

                            hist_name = base_name.replace('.pth', '_history.pkl')
                            best_model_path = os.path.join(best_models_dir, base_name)
                            history_path = os.path.join(best_models_dir, hist_name)
                        else:
                            # Fallback if csv_path is unexpectedly None
                            prefix_star = "" if enable_edges else "star_"
                            if enable_second_degree:
                                prefix_star = "2nddegree_" + prefix_star
                            
                            prefix = f"best_lstm_{prefix_star}{product_id}_{metric}_res_{MODEL_TYPE}" if USE_RESIDUALS else f"best_lstm_{prefix_star}{product_id}_{metric}"
                            param_label = f"th_{current_threshold}" if is_threshold_mode else f"pct_{current_percentile}"
                            best_model_path = os.path.join(best_models_dir, f'{prefix}_{window_size}_{step_size}_{param_label}_seed_{seed}.pth')
                            history_path = os.path.join(best_models_dir, f'{prefix}_{window_size}_{step_size}_{param_label}_seed_{seed}_history.pkl')
                    else:
                        best_model_path = os.path.join(best_models_dir, f'best_lstm_{product_id}_no_emb_seed_{seed}.pth')
                        history_path = os.path.join(best_models_dir, f'best_lstm_{product_id}_no_emb_seed_{seed}_history.pkl')

                    print(f"Resolved LSTM checkpoint: {best_model_path}")
                    print(f"Resolved LSTM history: {history_path}")

                    if os.path.exists(best_model_path) and os.path.exists(history_path):
                        print(f"Loading existing model from {best_model_path}...")
                        model.load_state_dict(torch.load(best_model_path))
                        with open(history_path, 'rb') as f:
                            history = pickle.load(f)
                            train_losses = history['train_losses']
                            val_losses = history['val_losses']
                    else:
                        print(f"Model not found. Expected model at {best_model_path}")
                        print(f"Expected history at {history_path}")
                        print("Training new model...")
                        model, train_losses, val_losses, best_epoch, train_time = train_model(
                            seed=seed, epochs=EPOCHS, model=model, 
                            train_loader=train_loader, val_loader=val_loader, 
                            exog_cols=EXOG_COLS, criterion=criterion, criterion2=criterion2, 
                            optimizer=optimizer, device=device, 
                            best_model_path=best_model_path if SAVE_MODELS else None, scheduler=scheduler, patience=PATIENCE
                        )
                        
                        if SAVE_MODELS:
                            with open(history_path, 'wb') as f:
                                pickle.dump({
                                    'train_losses': train_losses, 'val_losses': val_losses,
                                    'best_epoch': best_epoch, 'train_time': train_time
                                }, f)

                    # Explicitly load the best saved model before inference to guarantee clean pipeline state
                    if SAVE_MODELS and os.path.exists(best_model_path):
                        print(f"Loading best weights from {best_model_path} for inference...")
                        model.load_state_dict(torch.load(best_model_path))

                    exog_test_data = df[EXOG_COLS][test_slice].values

                    inf_threshold = fixed_threshold if use_embeddings and fixed_threshold is not None else None

                    print("Running Inference...")
                    
                    distance_metrics = ['euclidean','manhattan', 'hamming', 'amplitude_offset', 'slope_consistency', 'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
                    current_df_wide = df_wide_scaled if metric in distance_metrics else df_wide_global
                    
                    forecast, inference_time = graph2vec_inference(
                        metric=metric, window_size=window_size, step_size=step_size,
                        model=model,
                        df=df, df_wide=current_df_wide, cat_labels=cat_labels_dict, date_col=DATE_COL,
                        scaler=scaler, exog_scaler=exog_scaler,
                        test_start_idx=test_start_idx, seq_length=seq_length,
                        forecast_window=forecast_horizon, device=device,
                        item_id=product_id, store_id=store_id, seed=seed,
                        criterion="MSELoss", val_scaled=val_scaled, test_scaled=test_scaled,
                        exog_val_scaled=exog_val_scaled, exog_test_scaled=exog_test_scaled,
                        exog_test_raw=exog_test_data, exog_cols=EXOG_COLS,
                        product_scalers=product_scalers,
                        save_plot_path=None,
                        node_embeddings=aligned_embeddings if use_embeddings else None,
                        graph2vec_model=graph2vec_model if use_embeddings else None,
                        enable_edges_within_star=enable_edges,
                        enable_second_degree=enable_second_degree,
                        percentile=current_percentile if not is_threshold_mode else None,  # Passes percentile mode if used (otherwise None)
                        threshold=inf_threshold,  # Pass the threshold inferred from the percentile
                        create_plots=False  # We will create combined plots later, so disable individual plotting here
                    )

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
                
                    th_str = f"{inf_threshold:.4f}" if inf_threshold is not None else "N/A"
                    
                    if use_embeddings:
                        param_str_label = f"th:{current_threshold}" if is_threshold_mode else f"pct:{current_percentile} (val:{th_str})"
                        label_name = f"{param_str_label}|w:{window_size}|st:{step_size}|e:{enable_edges}|2nd:{enable_second_degree}"
                    else:
                        label_name = "No Embeddings"
                
                    if metric == 'no_emb':
                        base_forecast, base_train_losses, base_val_losses = forecast, train_losses, val_losses
                        base_rmse, base_mae, base_bias = rmse, mae, bias
                        base_score, base_pocid = score, pocid

                    results_by_w_s[key]['forecasts'][label_name] = forecast
                    results_by_w_s[key]['train_losses'][label_name] = train_losses
                    results_by_w_s[key]['val_losses'][label_name] = val_losses
                    results_by_w_s[key]['rmse'][label_name] = rmse
                    results_by_w_s[key]['mae'][label_name] = mae
                    results_by_w_s[key]['bias'][label_name] = bias
                    results_by_w_s[key]['score'][label_name] = score
                    results_by_w_s[key]['pocid'][label_name] = pocid
                
                    print(f"Finished {metric} @ {param_val} -> RMSE: {rmse:.4f}\n")
                    
                    # Append results to a persistent CSV file
                    import csv
                    csv_results_path = os.path.join(script_dir, f"{metric}.csv")
                    file_exists = os.path.exists(csv_results_path)
                    
                    with open(csv_results_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(["product_id", "store_id", "seed", "metric", "window_size", "step_size", "threshold", "percentile", "enable_edges", "enable_second_degree", "rmse", "mae", "bias", "r2_score", "pocid"])
                        
                        writer.writerow([
                            product_id, 
                            store_id, 
                            seed, 
                            metric, 
                            15 if metric == 'no_emb' else window_size, 
                            1 if metric == 'no_emb' else step_size, 
                            current_threshold if use_embeddings else "", 
                            current_percentile if use_embeddings else "", 
                            enable_edges if use_embeddings else "", 
                            enable_second_degree if use_embeddings else "", 
                            rmse, 
                            mae, 
                            bias, 
                            score, 
                            pocid
                        ])

                train_index = df[DATE_COL][train_slice].values
                val_index = df[DATE_COL][val_slice].values
                test_index = df[DATE_COL][test_slice].values

                if metric == 'no_emb':
                    values_str = "no_thresholds"
                    sub_dir = os.path.join(grid_search_plots_dir, 'no_emb', f'window_{15}', f'step_{1}', f'item_{product_id}', values_str)
                    os.makedirs(sub_dir, exist_ok=True)
                    save_plot_path = os.path.join(sub_dir, f"item_{product_id}_store_{store_id}_no_emb_seed_{seed}.html")
                    emb_title = f'Baseline LSTM Forecast (No Embeddings | Seed={seed})'
                    
                    if SAVE_PLOTS:
                        print(f"Saving combined plot to: {os.path.abspath(save_plot_path)}")
                        plot_results(train, val, test, results_by_w_s[(None, 15, 1)]['forecasts'], train_index, val_index, test_index,
                                     results_by_w_s[(None, 15, 1)]['train_losses'], results_by_w_s[(None, 15, 1)]['val_losses'], metric=metric, embedding_strategy='graph2vec',
                                     window_size=15, step_size=1, threshold=None, percentile=None,
                                     target_col=TARGET_COL, title=f'{emb_title} (Item={product_id})', seed=seed,
                                     save_path=save_plot_path, rmse=results_by_w_s[(None, 15, 1)]['rmse'], mae=results_by_w_s[(None, 15, 1)]['mae'], 
                                     bias=results_by_w_s[(None, 15, 1)]['bias'], score=results_by_w_s[(None, 15, 1)]['score'], pocid=results_by_w_s[(None, 15, 1)]['pocid'])
                else:
                    metric_type = infer_metric_type(metric)
                    
                    # Group by window and step to combine all thresholds in a single plot
                    grouped_results = {}
                    for (p, w, s), res_dicts in results_by_w_s.items():
                        key = (w, s)
                        if key not in grouped_results:
                            grouped_results[key] = {
                                'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                                'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {}
                            }
                        grouped_results[key]['forecasts'].update(res_dicts['forecasts'])
                        grouped_results[key]['train_losses'].update(res_dicts['train_losses'])
                        grouped_results[key]['val_losses'].update(res_dicts['val_losses'])
                        grouped_results[key]['rmse'].update(res_dicts['rmse'])
                        grouped_results[key]['mae'].update(res_dicts['mae'])
                        grouped_results[key]['bias'].update(res_dicts['bias'])
                        grouped_results[key]['score'].update(res_dicts['score'])
                        grouped_results[key]['pocid'].update(res_dicts['pocid'])

                    import hashlib
                    for (w, s), res_dicts in grouped_results.items():
                        if thresholds is not None and len(thresholds) > 0 and percentiles is None:
                            raw_str = "_".join(map(str, thresholds))
                        else:
                            raw_str = "_".join(map(str, percentiles))
                        
                        # Hash the configuration to avoid Extremely Long Path issues
                        values_str = hashlib.md5(raw_str.encode()).hexdigest()[:8]
                        
                        sub_dir = os.path.join(grid_search_plots_dir, metric_type, f'window_{w}', f'step_{s}', f'item_{product_id}', values_str)
                        os.makedirs(sub_dir, exist_ok=True)
                        
                        # Shorten the filename to avoid Windows MAX_PATH (260 chars) limitation
                        save_plot_path = os.path.join(sub_dir, f"item_{product_id}_{metric}_seed_{seed}_all_configs.html")
                        emb_title = f'Graph2Vec Forecasts ({metric} | Seed={seed} | W={w} | S={s})'

                        if SAVE_PLOTS:
                            print(f"Saving combined plot to: {os.path.abspath(save_plot_path)}")
                            plot_results(train, val, test, res_dicts['forecasts'], train_index, val_index, test_index,
                                         res_dicts['train_losses'], res_dicts['val_losses'], metric=metric, embedding_strategy='graph2vec',
                                         window_size=w, step_size=s, threshold=None, percentile=None,
                                         target_col=TARGET_COL, title=f'{emb_title} (Item={product_id})', seed=seed,
                                         save_path=save_plot_path, rmse=res_dicts['rmse'], mae=res_dicts['mae'], 
                                         bias=res_dicts['bias'], score=res_dicts['score'], pocid=res_dicts['pocid'])
                                     
    # Generate correlation plots at the very end
    if SAVE_PLOTS:
        import matplotlib.pyplot as plt
        print("\nGenerating Correlation Plots across all collected CSVs...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for csv_file in os.listdir(script_dir):
        if csv_file.endswith('.csv') and csv_file != 'no_emb.csv':
            metric_name = csv_file.replace('.csv', '')
            csv_path = os.path.join(script_dir, csv_file)
            
            try:
                res_df = pd.read_csv(csv_path)
                # Plot setup
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(f'Threshold vs RMSE and MAE | Metric: {metric_name}', fontsize=16)

                # Need to check if they used direct 'threshold' or 'percentile'
                x_col = 'threshold' if res_df['threshold'].notna().any() else 'percentile'
                
                # Drop rows where x_col might be missing
                plot_data = res_df.dropna(subset=[x_col, 'rmse', 'mae']).sort_values(by=x_col)
                
                if plot_data.empty:
                    continue

                ax1.plot(plot_data[x_col], plot_data['rmse'], marker='o', linestyle='-', color='b')
                ax1.set_title(f'{x_col.capitalize()} vs RMSE')
                ax1.set_xlabel(x_col.capitalize())
                ax1.set_ylabel('RMSE')
                ax1.grid(True)

                ax2.plot(plot_data[x_col], plot_data['mae'], marker='s', linestyle='-', color='r')
                ax2.set_title(f'{x_col.capitalize()} vs MAE')
                ax2.set_xlabel(x_col.capitalize())
                ax2.set_ylabel('MAE')
                ax2.grid(True)

                plot_save_path = os.path.join(script_dir, f"{metric_name}_correlation_plot.png")
                plt.tight_layout()
                plt.savefig(plot_save_path)
                plt.close()
                
                print(f"Saved correlation plot for {metric_name} at {plot_save_path}")
            except Exception as e:
                print(f"Failed to generate plot for {csv_file}: {e}")

if __name__ == '__main__':
    main()
