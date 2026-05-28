import os
import random
import sys
import pickle
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths & Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))                                  # repo root
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))                                        # DynamicSimilarities/
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'GraphAnalysis')))                       # for neighbourhood_graph
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'Graph2vec_FixedThreshold', 'LSTM')))    # for generate_graph2vecwithadaptativethreshold + plots

from model_utils.utils import generate_exogenous_features, compute_metrics
from plots import plot_results  # from sibling Graph2vec_FixedThreshold/LSTM/plots.py

from utils import neighbourhood_graph, compute_distances_1vsAll, compute_similarities_1vsAll  # GraphAnalysis/utils.py

# Local GCN + LSTM modules
from gcn_lstm_dataset import (
    GCNTimeSeriesDataset,
    collate_pyg_ts,
    build_pyg_graphs_from_nx_windows,
    _window_node_features,
)
from gcn_lstm_model import SimpleGCNLSTMForecaster
from train import train_model
DISTANCE_METRICS = {
    'euclidean', 'hamming', 'amplitude_offset', 'slope_consistency',
    'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss', 'manhattan', 'twed', 'erp', 'stid'
}
SIMILARITY_METRICS = {'pearson', 'spearman', 'kendall'}


def infer_metric_type(metric, metric_type=None):
    if metric_type is not None:
        if metric_type not in {'distance', 'similarity'}:
            raise ValueError("metric_type must be either 'distance' or 'similarity'")
        return metric_type
    if metric in DISTANCE_METRICS:
        return 'distance'
    if metric in SIMILARITY_METRICS:
        return 'similarity'
    raise ValueError(
        f"Metric {metric} not supported. "
        f"Distance metrics: {sorted(DISTANCE_METRICS)}; "
        f"similarity metrics: {sorted(SIMILARITY_METRICS)}"
    )
# Constants
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/data_andre.feather'))
DATE_COL = 'date'
TARGET_COL = 'value'
#SEEDS = [42, 1000, 26008, 213626, 907969, 5219788,13451285]  # Add more seeds as needed
SEEDS = [42, 1000, 26008, 213626]

# Add the products and stores you want to iterate over
PRODUCTS_TO_TEST = [
    (26008, 6269),
    (907967, 6269),
    (907969, 6269),
    (911753, 6269),
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

# Grid Search Parameters Setup
'''
grid_configs = [
    #{'metric': 'pearson', 'thresholds': [0.8,0.9, 0.95]},
    #{'metric': 'pearson', 'percentiles':  [0.5, 1, 2]},
    {'metric': 'spearman', 'thresholds': [round(t, 3) for t in np.arange(0.75, 0.85, 0.001)]},
    {'metric': 'pearson', 'thresholds': [round(t, 3) for t in np.arange(0.7, 0.82, 0.001)]},
    {'metric': 'kendall', 'thresholds': [round(t, 3) for t in np.arange(0.6, 0.74, 0.001)]},
    #{'metric': 'cid', 'thresholds': [round(t, 2) for t in np.arange(2, 3.1, 0.01)]},
]
'''
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
    {'metric': 'spearman', 'thresholds': [0.75,0.82,0.85,0.88,0.91]},
    #{'metric': 'cid', 'thresholds': [round(t, 2) for t in np.arange(2.0, 3.2, 0.01)]},
    # Distâncias Robustas e Lock-step
    #{'metric': 'amplitude_offset', 'thresholds': [round(t, 2) for t in np.arange(2.0, 3.5, 0.01)]},
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


# ──────────────────────────────────────────────────────────────────────────
# Helpers — GCN graph alignment + recursive inference
# ──────────────────────────────────────────────────────────────────────────
def _make_pad_graph(template: Data) -> Data:
    """Zero-valued single-node graph matching the feature width of ``template``."""
    in_feats = template.x.shape[1]
    return Data(
        x=torch.zeros(1, in_feats, dtype=torch.float32),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        edge_attr=torch.zeros(1, 1, dtype=torch.float32),
        num_nodes=1,
    )


def _align_pyg_windows_to_timeline(pyg_windows, window_size, step_size, T):
    """
    Convert the per-window PyG list (one Data per sliding window) into a
    per-day list of length ``T`` where ``aligned[t]`` is the graph whose
    window *ends* at day ``t``.  The first ``window_size - 1`` slots are
    filled with zero-graphs so day-0 indexing is safe.
    """
    if step_size != 1:
        raise NotImplementedError("Per-day alignment helper currently assumes step_size=1")

    pad = _make_pad_graph(pyg_windows[0])
    aligned = [pad] * (window_size - 1) + list(pyg_windows)
    if len(aligned) < T:
        aligned += [aligned[-1]] * (T - len(aligned))
    else:
        aligned = aligned[:T]
    return aligned


@torch.no_grad()
def _recursive_forecast_gcn(model, ts_seed, graph_seed, exog_test_scaled,
                            scaler, horizon, device):
    """
    Recursive (one-step-at-a-time) inference for the GCN+LSTM stack.

    Parameters
    ----------
    model            : SimpleGCNLSTMForecaster (eval mode handled here)
    ts_seed          : (L, 1+n_exog) np.ndarray  – last observed LSTM window in
                       scaled space (column 0 = scaled target). Row L-1 already
                       contains the exog for the FIRST predicted step.
    graph_seed       : ``Data`` ego-graph aligned to the last observed step.
                       Kept frozen across the rollout (no neighbour ground truth
                       in the future).
    exog_test_scaled : (horizon, n_exog) actual scaled exog for the test window.
                       Used to advance the exog row each step.
    scaler           : sklearn MinMaxScaler fit on the target.
    horizon          : number of recursive steps to roll out.
    device           : torch.device

    Returns
    -------
    np.ndarray (horizon,) — predictions in the original (un-scaled) target space.
    """
    model.eval()
    ts = np.asarray(ts_seed, dtype=np.float32).copy()
    graph = graph_seed.clone()
    preds_scaled = []

    for step in range(horizon):
        # advance exog of the last LSTM row to the step we are about to predict
        if step > 0 and exog_test_scaled is not None and ts.shape[1] > 1:
            ts[-1, 1:] = exog_test_scaled[step]

        ts_t = torch.from_numpy(ts).unsqueeze(0).to(device)
        batch = Batch.from_data_list([graph]).to(device)
        tidx = batch.ptr[:-1].to(device)

        out = model(batch, tidx, ts_t)                              # (1, H, 1)
        y_hat = float(out[0, -1, 0].detach().cpu().item())
        preds_scaled.append(y_hat)

        # roll LSTM window: shift left, append new target
        ts = np.vstack([ts[1:], ts[-1:].copy()])
        ts[-1, 0] = y_hat

    preds_scaled = np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()


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

        forecast_horizon = 153
        seq_length = 30
        train_size = 455
        val_size = 153
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

            # GCN+LSTM requires a graph for every config — no_emb baseline is dropped.
            all_configs = list(grid_configs)
            base_forecast, base_train_losses, base_val_losses = None, None, None
            base_rmse, base_mae, base_bias, base_score, base_pocid = None, None, None, None, None

            for config in all_configs:
                metric = config['metric']
                thresholds = config.get('thresholds', [None])
                percentiles = config.get('percentiles', [None])

                results_by_w_s = {}

                is_threshold_mode = thresholds is not None and thresholds != [None]
                params = thresholds if is_threshold_mode else percentiles
                iterator = itertools.product(params, window_sizes, step_sizes, enable_edges_opts, enable_second_degree_opts)

                for param_val, window_size, step_size, enable_edges, enable_second_degree in iterator:
                    use_embeddings = True

                    current_threshold = param_val if is_threshold_mode else None
                    current_percentile = param_val if not is_threshold_mode else None

                    key = (param_val, window_size, step_size)
                    if key not in results_by_w_s:
                        results_by_w_s[key] = {
                            'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                            'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {},
                            'threshold': None
                        }
                        # (GCN+LSTM has no separate no_emb baseline to carry over.)

                    print(f"\n{'='*60}")
                    param_str = f"threshold={current_threshold}" if is_threshold_mode else f"percentile={current_percentile}"
                    print(f"Running Experiment: metric={metric}, {param_str}, "
                          f"window_size={window_size}, enable_edges={enable_edges}, 2nd_degree={enable_second_degree}")
                    print(f"{'='*60}")

                    # ── 1. Build per-window NX graphs (target ego-graphs) ────────────
                    metric_type = infer_metric_type(metric)
                    distance_metrics = ['euclidean', 'manhattan', 'hamming', 'amplitude_offset',
                                        'slope_consistency', 'phase_invariance', 'dtw', 'cid',
                                        'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
                    current_df_wide = df_wide_scaled if metric in distance_metrics else df_wide_global
                    compute_func = compute_distances_1vsAll if metric_type == 'distance' else compute_similarities_1vsAll

                    nx_graphs, fixed_threshold = neighbourhood_graph(
                        product_id=product_id,
                        df=current_df_wide,
                        metric=metric,
                        metric_type=metric_type,
                        window_size=window_size,
                        compute_func=compute_func,
                        threshold=current_threshold if is_threshold_mode else None,
                        percentile=current_percentile if not is_threshold_mode else None,
                        step_size=step_size,
                        cat_labels=cat_labels_dict,
                        plot_dir=None,
                        residuals=USE_RESIDUALS,
                        enable_edges_within_star=enable_edges,
                        enable_second_degree=enable_second_degree,
                        train_end_idx=global_val_start_idx,
                    )
                    print(f"Resolved graph threshold={current_threshold}: {fixed_threshold}")
                    results_by_w_s[key]['threshold'] = fixed_threshold

                    # ── 2. Convert to per-window PyG ego-graphs, align to timeline ───
                    pyg_windows = build_pyg_graphs_from_nx_windows(
                        nx_graphs, current_df_wide, product_id,
                        window_size=window_size, step_size=step_size,
                    )
                    T_global = current_df_wide.shape[1]
                    pyg_aligned_global = _align_pyg_windows_to_timeline(
                        pyg_windows, window_size=window_size, step_size=step_size, T=T_global,
                    )
                    # Per-product timeline = global timeline (we already require full-length products)
                    # but offset into df_wide_global is needed for products that don't span the full range.
                    # Use the global indices we computed once at the top.
                    product_offset = T_global - len(df)
                    pyg_train = pyg_aligned_global[product_offset + train_start_idx : product_offset + val_start_idx]
                    pyg_val   = pyg_aligned_global[product_offset + val_start_idx   : product_offset + test_start_idx]
                    pyg_seed_graph = pyg_aligned_global[product_offset + test_start_idx - 1]

                    # ── 3. Datasets / loaders (GCN-aware) ────────────────────────────
                    use_pin_memory = torch.cuda.is_available()
                    train_dataset = GCNTimeSeriesDataset(
                        target_data=train_scaled,
                        exog_data=exog_train_scaled if EXOG_COLS else None,
                        seq_length=seq_length,
                        pyg_graphs=pyg_train,
                        graph_window_size=window_size,
                    )
                    train_loader = DataLoader(
                        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=use_pin_memory, collate_fn=collate_pyg_ts,
                    )
                    val_dataset = GCNTimeSeriesDataset(
                        target_data=val_scaled,
                        exog_data=exog_val_scaled if EXOG_COLS else None,
                        seq_length=seq_length,
                        pyg_graphs=pyg_val,
                        graph_window_size=window_size,
                    )
                    val_loader = DataLoader(
                        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=use_pin_memory, collate_fn=collate_pyg_ts,
                    )

                    # ── 4. Model + optimiser (re-seeded for determinism) ─────────────
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)

                    in_channels = pyg_train[0].x.shape[1]  # 8 from _window_node_features
                    lstm_input_size = 1 + (len(EXOG_COLS) if EXOG_COLS else 0)
                    model = SimpleGCNLSTMForecaster(
                        in_channels=in_channels,
                        gcn_hidden=HIDDEN_SIZE,
                        d_g=16,
                        lstm_input_size=lstm_input_size,
                        lstm_hidden=HIDDEN_SIZE,
                        lstm_layers=NUM_LAYERS,
                        horizon=1,
                        dropout=DROPOUT,
                    ).to(device)
                    criterion = nn.MSELoss()
                    criterion2 = nn.MSELoss()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=PATIENCE // 3,
                    )

                    # ── 5. Checkpoint paths ──────────────────────────────────────────
                    model_dir_label = f"th{current_threshold}" if is_threshold_mode else f"pct{current_percentile}"
                    best_models_dir = os.path.join(
                        best_models_seed_dir, str(window_size), str(step_size), metric, model_dir_label,
                    )
                    os.makedirs(best_models_dir, exist_ok=True)

                    prefix_star = "" if enable_edges else "star_"
                    if enable_second_degree:
                        prefix_star = "2nddegree_" + prefix_star
                    res_tag = f"_res_{MODEL_TYPE}" if USE_RESIDUALS else ""
                    param_label = f"th_{current_threshold}" if is_threshold_mode else f"pct_{current_percentile}"
                    base_name = (f"best_gcnlstm_{prefix_star}{product_id}_{metric}"
                                 f"_w{window_size}_s{step_size}_{param_label}{res_tag}_seed_{seed}")
                    best_model_path = os.path.join(best_models_dir, f"{base_name}.pth")
                    history_path = os.path.join(best_models_dir, f"{base_name}_history.pkl")

                    print(f"Resolved checkpoint: {best_model_path}")

                    # ── 6. Train (or reload) ─────────────────────────────────────────
                    if os.path.exists(best_model_path) and os.path.exists(history_path):
                        print(f"Loading existing model from {best_model_path}...")
                        model.load_state_dict(torch.load(best_model_path, map_location=device))
                        with open(history_path, 'rb') as f:
                            history = pickle.load(f)
                            train_losses = history['train_losses']
                            val_losses = history['val_losses']
                    else:
                        print("Training new GCN+LSTM model...")
                        model, train_losses, val_losses, best_epoch, train_time = train_model(
                            seed=seed, epochs=EPOCHS, model=model,
                            train_loader=train_loader, val_loader=val_loader,
                            criterion=criterion, criterion2=criterion2,
                            optimizer=optimizer, device=device,
                            best_model_path=best_model_path if SAVE_MODELS else None,
                            scheduler=scheduler, patience=PATIENCE,
                        )
                        if SAVE_MODELS:
                            with open(history_path, 'wb') as f:
                                pickle.dump({
                                    'train_losses': train_losses, 'val_losses': val_losses,
                                    'best_epoch': best_epoch, 'train_time': train_time,
                                }, f)

                    if SAVE_MODELS and os.path.exists(best_model_path):
                        print(f"Loading best weights from {best_model_path} for inference...")
                        model.load_state_dict(torch.load(best_model_path, map_location=device))

                    # ── 7. Recursive inference ───────────────────────────────────────
                    inf_threshold = fixed_threshold
                    print("Running Inference...")

                    # Build LSTM seed window: last seq_length target values of val,
                    # exog rows aligned so last row holds exog_test[0] (the first step we predict).
                    if EXOG_COLS:
                        exog_seed_rows = np.vstack([
                            exog_val_scaled[-(seq_length - 1):],
                            exog_test_scaled[0:1],
                        ])
                        ts_seed = np.column_stack([
                            val_scaled[-seq_length:].reshape(-1, 1),
                            exog_seed_rows,
                        ]).astype(np.float32)
                    else:
                        ts_seed = val_scaled[-seq_length:].reshape(-1, 1).astype(np.float32)

                    forecast = _recursive_forecast_gcn(
                        model=model,
                        ts_seed=ts_seed,
                        graph_seed=pyg_seed_graph,
                        exog_test_scaled=exog_test_scaled if EXOG_COLS else None,
                        scaler=scaler,
                        horizon=forecast_horizon,
                        device=device,
                    )

                    valid_mask = ~np.isnan(forecast)
                    valid_test = test[valid_mask]
                    valid_forecast = np.array(forecast)[valid_mask]

                    rmse, mae, bias, score, pocid = None, None, None, None, None
                    try:
                        rmse, mae, bias, score, pocid = compute_metrics(valid_test, valid_forecast)
                    except Exception:
                        if len(valid_test) > 0:
                            rmse = float(np.sqrt(mean_squared_error(valid_test, valid_forecast)))
                            mae = float(mean_absolute_error(valid_test, valid_forecast))
                            bias = float(np.mean(valid_forecast - valid_test))
                            score = float(r2_score(valid_test, valid_forecast))

                    th_str = f"{inf_threshold:.4f}" if inf_threshold is not None else "N/A"
                    param_str_label = (f"th:{current_threshold}" if is_threshold_mode
                                       else f"pct:{current_percentile} (val:{th_str})")
                    label_name = (f"{param_str_label}|w:{window_size}|st:{step_size}"
                                  f"|e:{enable_edges}|2nd:{enable_second_degree}")

                    results_by_w_s[key]['forecasts'][label_name] = forecast
                    results_by_w_s[key]['train_losses'][label_name] = train_losses
                    results_by_w_s[key]['val_losses'][label_name] = val_losses
                    results_by_w_s[key]['rmse'][label_name] = rmse
                    results_by_w_s[key]['mae'][label_name] = mae
                    results_by_w_s[key]['bias'][label_name] = bias
                    results_by_w_s[key]['score'][label_name] = score
                    results_by_w_s[key]['pocid'][label_name] = pocid

                    print(f"Finished {metric} @ {param_val} -> RMSE: {rmse}\n")

                    # Append results to a persistent CSV file
                    import csv
                    csv_results_path = os.path.join(script_dir, f"{metric}.csv")
                    file_exists = os.path.exists(csv_results_path)

                    with open(csv_results_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(["product_id", "store_id", "seed", "metric",
                                             "window_size", "step_size", "threshold", "percentile",
                                             "enable_edges", "enable_second_degree",
                                             "rmse", "mae", "bias", "r2_score", "pocid"])

                        writer.writerow([
                            product_id,
                            store_id,
                            seed,
                            metric,
                            window_size,
                            step_size,
                            current_threshold if current_threshold is not None else "",
                            current_percentile if current_percentile is not None else "",
                            enable_edges,
                            enable_second_degree,
                            rmse,
                            mae,
                            bias,
                            score,
                            pocid,
                        ])

                train_index = df[DATE_COL][train_slice].values
                val_index = df[DATE_COL][val_slice].values
                test_index = df[DATE_COL][test_slice].values

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
                    emb_title = f'GCN+LSTM Forecasts ({metric} | Seed={seed} | W={w} | S={s})'

                    if SAVE_PLOTS:
                        print(f"Saving combined plot to: {os.path.abspath(save_plot_path)}")
                        plot_results(train, val, test, res_dicts['forecasts'], train_index, val_index, test_index,
                                     res_dicts['train_losses'], res_dicts['val_losses'], metric=metric, embedding_strategy='gcn',
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
