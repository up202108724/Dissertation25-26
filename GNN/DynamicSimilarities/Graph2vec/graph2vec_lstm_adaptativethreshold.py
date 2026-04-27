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
from graph2vecdataset import TimeSeriesDataset
from model_utils.utils import generate_exogenous_features, compute_metrics 
from model_utils.plots import plot_results #ensure available
from GNN.DynamicSimilarities.Graph2vec.generate_graph2vecwithadaptativethreshold import load_or_generate_embeddings
from train import train_model
from graph2vecinference_adaptativethreshold import graph2vec_inference

# Constants
# Resolve DATA_PATH perfectly from the file directory upwards
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/data_andre.feather'))
DATE_COL = 'date'
TARGET_COL = 'value'
SEEDS = [42]  # Add more seeds as needed

# Add the products and stores you want to iterate over
PRODUCTS_TO_TEST = [
  (26008,6269),
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

            # Grid Search Parameters Setup
            metrics = ['cid']
            percentiles = [0.5,1,2]
            window_sizes = [15]     
            step_sizes = [1]
            enable_edges_opts = [True]
            enable_second_degree_opts = [False]
            p_cutoff= 90# We will keep this False for the main analysis, but you can set to True to include second-degree neighbors in the graph construction
            adaptive_strategies = [None, 'Mean__kStd','Median + k*MAD',f'Percentile {p_cutoff}']  # Add more strategies as needed
            USE_RESIDUALS = False
            MODEL_TYPE = 'ridge'
            EPOCHS = 1000
            PATIENCE = 100
            LEARNING_RATE = 0.001
            HIDDEN_SIZE = 32
            NUM_LAYERS = 1
            DROPOUT = 0.0

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Get the directory where `graph2vec_lstm.py` is located to anchor our save paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            grid_search_plots_dir = os.path.join(script_dir, 'grid_search_plots', f'seed_{seed}')
            best_models_dir = os.path.join(script_dir, 'best_models', f'seed_{seed}')
            
            os.makedirs(grid_search_plots_dir, exist_ok=True)
            os.makedirs(best_models_dir, exist_ok=True)

            all_metrics = ['no_emb'] + metrics
            base_forecast, base_train_losses, base_val_losses = None, None, None
            base_rmse, base_mae, base_bias, base_score, base_pocid = None, None, None, None, None

            for metric in all_metrics:
                forecasts_dict, train_losses_dict, val_losses_dict = {}, {}, {}
                rmse_dict, mae_dict, bias_dict, score_dict, pocid_dict = {}, {}, {}, {}, {}

                # If we already have the baseline and we are plotting embeddings, add it!
                if metric != 'no_emb' and base_forecast is not None:
                    forecasts_dict["No Embeddings"] = base_forecast
                    train_losses_dict["No Embeddings"] = base_train_losses
                    val_losses_dict["No Embeddings"] = base_val_losses
                    rmse_dict["No Embeddings"] = base_rmse
                    mae_dict["No Embeddings"] = base_mae
                    bias_dict["No Embeddings"] = base_bias
                    score_dict["No Embeddings"] = base_score
                    pocid_dict["No Embeddings"] = base_pocid

                if metric == 'no_emb':
                    iterator = [(None, 15, 1, False, False, None)]
                else:
                    iterator = itertools.product(percentiles, window_sizes, step_sizes, enable_edges_opts, enable_second_degree_opts, adaptive_strategies)
            
                for percentile, window_size, step_size, enable_edges, enable_second_degree, adaptive_strategy in iterator:
                    use_embeddings = (metric != 'no_emb')

                    print(f"\n{'='*60}")
                    if use_embeddings:
                        print(f"Running Experiment: metric={metric}, percentile={percentile}, window_size={window_size}, enable_edges={enable_edges}, 2nd_degree={enable_second_degree}, adapt={adaptive_strategy}")
                    else:
                        print("Running Experiment: BASELINE (no graph embeddings)")
                    print(f"{'='*60}")

                    if use_embeddings:
                        graph_embeddings, graph2vec_model, csv_path = load_or_generate_embeddings(
                            product_id=product_id,
                            metric=metric,
                            window_size=window_size,
                            step_size=step_size,
                            threshold=None,
                            enable_edges_within_star=enable_edges,
                            enable_second_degree=enable_second_degree,
                            percentile=percentile,
                            use_residuals=USE_RESIDUALS,
                            model_type=MODEL_TYPE,
                            seed=seed,
                            adaptive_strategy=adaptive_strategy
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

                    # Initialize Model
                    model = LSTM(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
                    criterion = nn.MSELoss()
                    criterion2 = nn.MSELoss()  
                    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE//3)

                    if use_embeddings:
                        if csv_path:
                            # Base the LSTM model name strictly on the embeddings filename
                            csv_basename = os.path.basename(csv_path)
                            if adaptive_strategy:
                                import re
                                safe_strat = re.sub(r'[^a-zA-Z0-9_\-]', '', adaptive_strategy.replace(' ', '_'))
                                csv_basename = f"adapt_{safe_strat}_" + csv_basename

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
                            if adaptive_strategy:
                                import re
                                safe_strat = re.sub(r'[^a-zA-Z0-9_\-]', '', adaptive_strategy.replace(' ', '_'))
                                prefix_star = f"adapt_{safe_strat}_{prefix_star}"
                            prefix = f"best_lstm_{prefix_star}{product_id}_{metric}_res_{MODEL_TYPE}" if USE_RESIDUALS else f"best_lstm_{prefix_star}{product_id}_{metric}"
                            best_model_path = os.path.join(best_models_dir, f'{prefix}_{window_size}_{step_size}_percentile_{percentile}_seed_{seed}.pth')
                            history_path = os.path.join(best_models_dir, f'{prefix}_{window_size}_{step_size}_percentile_{percentile}_seed_{seed}_history.pkl')
                    else:
                        best_model_path = os.path.join(best_models_dir, f'best_lstm_{product_id}_no_emb_seed_{seed}.pth')
                        history_path = os.path.join(best_models_dir, f'best_lstm_{product_id}_no_emb_seed_{seed}_history.pkl')

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
                    
                    historical_weights = None
                    if use_embeddings and adaptive_strategy is not None:
                        # Load dynamic graphs PKL (the NON-adaptive threshold one) to get historical weights
                        prefix_p = "" if enable_edges else "star_"
                        if enable_second_degree: prefix_p = "2nddegree_" + prefix_p
                        
                        base_pkl_filename = f"{prefix_p}dynamic_graphs_{metric}_Window{window_size}_Step{step_size}_pct{percentile}.pkl"
                        
                        curr_dir_p = os.path.dirname(os.path.abspath(__file__))
                        dir_label_p = f"pct{percentile}"
                        plot_dir_name_p = f"{prefix_p}{dir_label_p}" if prefix_p else dir_label_p
                        
                        base_dir_p = os.path.join(curr_dir_p, '..', 'GraphAnalysis', 'DynamicGraphPkls', str(product_id), metric, str(window_size), str(step_size), plot_dir_name_p)
                        base_pkl_path = os.path.join(base_dir_p, base_pkl_filename)
                        
                        if os.path.exists(base_pkl_path):
                            print(f"Extracting historical weights from {base_pkl_path}...")
                            with open(base_pkl_path, 'rb') as f:
                                base_graphs = pickle.load(f)
                            historical_weights = []
                            import math
                            for g in base_graphs:
                                for u, v, data in g.edges(data=True):
                                    w = data.get('weight', 0)
                                    if not math.isnan(w):
                                        historical_weights.append(w)
                        else:
                            print(f"WARNING: Could not find base pkl {base_pkl_path} to extract historical weights! Using empty list.")
                            historical_weights = []

                    print("Running Inference...")
                    forecast, inference_time = graph2vec_inference(
                        metric=metric, window_size=window_size, step_size=step_size,
                        threshold=None, percentile=percentile, model=model,
                        df=df, df_wide=df_wide_global, cat_labels=cat_labels_dict, date_col=DATE_COL,
                        scaler=scaler, exog_scaler=exog_scaler,
                        test_start_idx=test_start_idx, seq_length=seq_length,
                        forecast_window=forecast_horizon, device=device,
                        item_id=product_id, store_id=store_id, seed=seed,
                        criterion="MSELoss", val_scaled=val_scaled, test_scaled=test_scaled,
                        exog_val_scaled=exog_val_scaled, exog_test_scaled=exog_test_scaled,
                        exog_test_raw=exog_test_data, exog_cols=EXOG_COLS,
                        save_plot_path=None,
                        node_embeddings=aligned_embeddings if use_embeddings else None,
                        graph2vec_model=graph2vec_model if use_embeddings else None,
                        enable_edges_within_star=enable_edges,
                        enable_second_degree=enable_second_degree,
                        historical_weights=historical_weights,
                        adaptive_strategy=adaptive_strategy
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
                
                    if use_embeddings:
                        label_name = f"pct:{percentile}|w:{window_size}|st:{step_size}|edge:{enable_edges}|2nd:{enable_second_degree}"
                        if adaptive_strategy:
                            label_name += f"|Adapt:{adaptive_strategy}"
                    else:
                        label_name = "No Embeddings"
                
                    if metric == 'no_emb':
                        base_forecast, base_train_losses, base_val_losses = forecast, train_losses, val_losses
                        base_rmse, base_mae, base_bias = rmse, mae, bias
                        base_score, base_pocid = score, pocid

                    forecasts_dict[label_name] = forecast
                    train_losses_dict[label_name] = train_losses
                    val_losses_dict[label_name] = val_losses
                    rmse_dict[label_name] = rmse
                    mae_dict[label_name] = mae
                    bias_dict[label_name] = bias
                    score_dict[label_name] = score
                    pocid_dict[label_name] = pocid
                
                    print(f"Finished {metric} @ {percentile} -> RMSE: {rmse:.4f}\n")

                train_index = df[DATE_COL][train_slice].values
                val_index = df[DATE_COL][val_slice].values
                test_index = df[DATE_COL][test_slice].values
            
                if metric != 'no_emb':
                    save_plot_path = os.path.join(grid_search_plots_dir, f"item_{product_id}_store_{store_id}_{metric}_all_params_seed_{seed}.html")
                    emb_title = f'Graph2Vec Forecasts ({metric} | Seed={seed})'
                else:
                    save_plot_path = os.path.join(grid_search_plots_dir, f"item_{product_id}_store_{store_id}_no_emb_seed_{seed}.html")
                    emb_title = f'Baseline LSTM Forecast (No Embeddings | Seed={seed})'

                print(f"Saving combined plot to: {os.path.abspath(save_plot_path)}")
                plot_results(train, val, test, forecasts_dict, train_index, val_index, test_index,
                             train_losses_dict, val_losses_dict, metric=metric, embedding_strategy='graph2vec',
                             target_col=TARGET_COL, title=f'{emb_title} (Item={product_id})', seed=seed,
                             save_path=save_plot_path, rmse=rmse_dict, mae=mae_dict, bias=bias_dict, score=score_dict, pocid=pocid_dict)

if __name__ == '__main__':
    main()
