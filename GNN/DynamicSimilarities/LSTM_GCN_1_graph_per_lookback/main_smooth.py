"""
Grid-search runner for the PER-STEP GCN + LSTM forecaster
(one ego-graph per lookback day).

Mirrors ``GNN/DynamicSimilarities/LSTM_GCN/main.py`` (single-graph variant)
but adapted to the per-step model: the recursive-inference seed is a
deque of L ego-graphs, and an aligned sequence of per-day ``future_graphs``
is fed to the rollout so each forecast step sees the correct graph for the
day it is predicting.
"""

import os
import random
import sys
import pickle
import itertools
import time
import csv as _csv_mod
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Paths & sys.path setup (same convention as the single-graph runner) ────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))                                  # repo root
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))                                        # DynamicSimilarities/
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'GraphAnalysis')))                       # for neighbourhood_graph
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'Graph2vec_FixedThreshold', 'LSTM')))    # for plots

from inference import _recursive_forecast_gcn_perstep
from model_utils.utils import generate_exogenous_features, compute_metrics
from plots import plot_results  # from sibling Graph2vec_FixedThreshold/LSTM/plots.py

from utils import neighbourhood_graph, compute_distances_1vsAll, compute_similarities_1vsAll  # GraphAnalysis/utils.py

# Local (per-step) GCN+LSTM modules
from gcn_lstm_dataset import (
    GCNTimeSeriesDataset,
    collate_pyg_ts,
    build_pyg_graphs_from_nx_windows,
)
from gcn_lstm_model import SimpleGCNLSTMForecaster
from train import train_model


# ── Metric typing ──────────────────────────────────────────────────────────
DISTANCE_METRICS = {
    'euclidean', 'hamming', 'amplitude_offset', 'slope_consistency',
    'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss',
    'manhattan', 'twed', 'erp', 'stid',
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


# ── Constants (same defaults as LSTM_GCN/main.py) ──────────────────────────
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/df_smooth.feather'))
DATE_COL = 'date'
TARGET_COL = 'value'

SEEDS = [42]

# Leave as None to run all (item_id, store_id) pairs found in DATA_PATH.
# Override with a list of tuples to run only a subset, e.g.:
#   PRODUCTS_TO_TEST = [(26008, 6269), (907967, 6269)]
PRODUCTS_TO_TEST = None

EXOG_COLS = [
    "day_of_week", "day_of_month", "week_of_year", "week_of_month",
    "month", "quarter", "is_weekend",
    #"lag_1", "lag_7",  "lag_30",
    #"rolling_mean_excl_7", "rolling_mean_excl_3", "rolling_mean_excl_5","rolling_mean_excl_15",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_monday", "is_friday",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_pre_holiday_1", "is_pre_holiday_2", "is_pre_holiday_3", "is_pre_holiday_7",
    "is_post_holiday_1", "is_post_holiday_2", "is_post_holiday_3", "is_post_holiday_7",
    "is_bridge_day",
]
grid_configs = [
    {'metric': 'spearman', 'thresholds': [0.70, 0.85, 0.95, 0.99]},
]

window_sizes              = [15]
step_sizes                = [1]
enable_edges_opts         = [True]
enable_second_degree_opts = [False]
USE_RESIDUALS  = False
MODEL_TYPE     = 'ridge'
EPOCHS         = 1000
PATIENCE       = 100
LEARNING_RATE  = 0.001
HIDDEN_SIZE    = 32
NUM_LAYERS     = 1
DROPOUT        = 0.0
D_G            = 16          # per-step graph embedding dim
SAVE_MODELS    = False
SAVE_PLOTS     = True
USE_EMBEDDINGS = True
SAVE_EMBEDDINGS = False
GCN_NODE_FEATURES = 8           # output width of _window_node_features(); used for ablation dummy graphs

# ── Diagnostic switches ───────────────────────────────────────────────────
# Grid of ablation modes to run in a single execution:
#   True  — GCN z-embedding zeroed (ablation)
#   False — full GCN+LSTM model
ABLATE_Z_VALUES = [True, False]
# Per-epoch diagnostics (||z||, Var(z), ‖∇gcn‖/‖∇lstm‖) are appended to
# this CSV.  Suffix "_ablation" is added automatically when ABLATE_Z=True.
DIAG_CSV_NAME  = "diagnostics.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────────────────────────────────────────────────────
# Per-day alignment helper
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
    Convert the per-window PyG list (one Data per sliding window of width W)
    into a per-day list of length T where ``aligned[t]`` is the graph built
    *strictly before* day t (days t-W .. t-1).  This matches the
    GCNTimeSeriesDataset / Graph2Vec convention and is the pad that
    GUARANTEES no label leakage in the per-step variant (the graph chosen
    for the last lookback step never includes the label day).
    """
    if step_size != 1:
        raise NotImplementedError("Per-day alignment helper currently assumes step_size=1")

    pad = _make_pad_graph(pyg_windows[0])
    aligned = [pad] * window_size + list(pyg_windows)       # pad by W (not W-1)
    if len(aligned) < T:
        aligned += [aligned[-1]] * (T - len(aligned))
    else:
        aligned = aligned[:T]
    return aligned


# Recursive inference is imported from inference.py — the local duplicate
# was removed so that the leakage-safe (dynamic-lag) version is used.

# Exog columns whose value at day t is value[t-k] — must be recomputed from
# the model's own rolling forecast during recursive inference, otherwise the
# test-time series is contaminated with ground-truth lags from step >= 1.
LAG_K_BY_NAME = {"lag_1": 1, "lag_7": 7, "lag_30": 30}

# Any EXOG column whose name starts with this prefix is treated as a
# rolling_mean_excl_W feature (mean of the W RAW target values strictly
# preceding the day being predicted).  Same leakage profile as lags: must be
# recomputed from the rolling history during recursive inference.
ROLLING_MEAN_EXCL_PREFIX = "rolling_mean_excl_"


# ──────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────
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
    df = df.reset_index(drop=True)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, 'item_id', 'store_id']).reset_index(drop=True)
    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS)
    full_df = df.copy()

    # Build the list of (item_id, store_id) pairs to iterate over
    global PRODUCTS_TO_TEST
    if PRODUCTS_TO_TEST is None:
        PRODUCTS_TO_TEST = (
            full_df[["item_id", "store_id"]]
            .drop_duplicates()
            .sort_values(["item_id", "store_id"])
            .apply(lambda r: (int(r["item_id"]), int(r["store_id"])), axis=1)
            .tolist()
        )
        print(f"Running all {len(PRODUCTS_TO_TEST)} (item_id, store_id) pairs from dataset.")

    cat_labels_dict = (
        full_df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()
        if 'cat_label' in full_df.columns else {}
    )
    df_wide_global = full_df.pivot_table(
        index='item_id', columns=DATE_COL, values=TARGET_COL, aggfunc='sum'
    ).fillna(0)
    df_wide_global.columns = pd.to_datetime(df_wide_global.columns).strftime('%Y-%m-%d')

    Lcols = len(df_wide_global.columns)
    forecast_horizon_global = 153
    val_size_global         = 153
    train_size_global       = 455
    global_train_start_idx = max(0, Lcols - forecast_horizon_global - val_size_global - train_size_global)
    global_val_start_idx   = Lcols - forecast_horizon_global - val_size_global

    # per-product StandardScaler fit on training window, applied to whole history
    product_scalers = {}
    train_df_wide = df_wide_global.iloc[:, global_train_start_idx:global_val_start_idx]
    df_wide_scaled = df_wide_global.copy()
    for item_id_iter in df_wide_global.index:
        z_scaler = StandardScaler()
        z_scaler.fit(train_df_wide.loc[item_id_iter].values.reshape(-1, 1))
        product_scalers[item_id_iter] = z_scaler
        df_wide_scaled.loc[item_id_iter] = z_scaler.transform(
            df_wide_global.loc[item_id_iter].values.reshape(-1, 1)
        ).flatten()

    # ── Timing bookkeeping ───────────────────────────────────────────────
    timings_csv_path = os.path.join(SCRIPT_DIR, "timings.csv")
    seed_times = {}                  # seed -> cumulative seconds across products
    product_seed_times = []          # (product_id, store_id, seed, seconds)
    total_t0 = time.time()

    for product_id, store_id in PRODUCTS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"PROCESSING PRODUCT {product_id} FOR STORE {store_id}")
        print(f"{'='*80}\n")

        df_p = (
            full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)]
            .sort_values(DATE_COL).reset_index(drop=True)
        )

        forecast_horizon = 153
        seq_length       = 30
        train_size       = 455
        val_size         = 153
        BATCH_SIZE       = 32

        required_rows = forecast_horizon + val_size + train_size
        if len(df_p) < required_rows:
            print(f"Skipping Product {product_id} at Store {store_id}: "
                  f"Found {len(df_p)} rows, but {required_rows} are required.")
            continue

        test_start_idx  = len(df_p) - forecast_horizon
        val_start_idx   = test_start_idx - val_size
        train_start_idx = val_start_idx - train_size

        train_slice = slice(train_start_idx, val_start_idx)
        val_slice   = slice(val_start_idx,   test_start_idx)
        test_slice  = slice(test_start_idx,  None)

        train = df_p[TARGET_COL][train_slice].values
        val   = df_p[TARGET_COL][val_slice].values
        test  = df_p[TARGET_COL][test_slice].values

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
        val_scaled   = scaler.transform(val.reshape(-1, 1)).flatten()
        test_scaled  = scaler.transform(test.reshape(-1, 1)).flatten()

        if EXOG_COLS:
            exog_scaler = MinMaxScaler()
            exog_train_scaled = exog_scaler.fit_transform(df_p[EXOG_COLS][train_slice].values)
            exog_val_scaled   = exog_scaler.transform(df_p[EXOG_COLS][val_slice].values)
            exog_test_scaled  = exog_scaler.transform(df_p[EXOG_COLS][test_slice].values)
        else:
            exog_train_scaled = exog_val_scaled = exog_test_scaled = None
            exog_scaler = None

        for seed in SEEDS:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            print(f"\n--- RUNNING WITH SEED {seed} ---\n")
            seed_t0 = time.time()

            grid_search_plots_dir = os.path.join(SCRIPT_DIR, 'grid_search_plots', f'seed_{seed}')
            best_models_seed_dir  = os.path.join(SCRIPT_DIR, 'best_models',       f'seed_{seed}')
            os.makedirs(grid_search_plots_dir, exist_ok=True)
            os.makedirs(best_models_seed_dir,  exist_ok=True)

            # GCN+LSTM requires a graph for every config — no_emb baseline is dropped.
            all_configs = list(grid_configs)

            for config in all_configs:
                metric      = config['metric']
                thresholds  = config.get('thresholds',  [None])
                percentiles = config.get('percentiles', [None])

                results_by_w_s = {}

                is_threshold_mode = thresholds is not None and thresholds != [None]
                params   = thresholds if is_threshold_mode else percentiles
                iterator = itertools.product(
                    ABLATE_Z_VALUES, params, window_sizes, step_sizes,
                    enable_edges_opts, enable_second_degree_opts,
                )

                for ablate_z, param_val, window_size, step_size, enable_edges, enable_second_degree in iterator:
                    # When ablating, z is zeroed so the threshold has no effect on
                    # the trained weights.  Skip all but the first threshold value.
                    if ablate_z and param_val != params[0]:
                        continue

                    current_threshold  = param_val if is_threshold_mode else None
                    current_percentile = param_val if not is_threshold_mode else None

                    key = (ablate_z, param_val, window_size, step_size)
                    if key not in results_by_w_s:
                        results_by_w_s[key] = {
                            'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                            'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {},
                            'threshold': None,
                        }

                    print(f"\n{'='*60}")
                    param_str = (f"threshold={current_threshold}" if is_threshold_mode
                                 else f"percentile={current_percentile}")
                    print(f"Running Experiment: ablate_z={ablate_z}, metric={metric}, {param_str}, "
                          f"window_size={window_size}, enable_edges={enable_edges}, "
                          f"2nd_degree={enable_second_degree}")
                    print(f"{'='*60}")

                    # ── 1 & 2. Graph pipeline (skipped for ablation) ─────────
                    metric_type = infer_metric_type(metric)
                    if ablate_z:
                        # GCN output is zeroed in forward(); building hundreds of
                        # sliding-window graphs would be pure waste.  Pass dummy
                        # single-node placeholder graphs instead.
                        fixed_threshold = None
                        results_by_w_s[key]['threshold'] = fixed_threshold
                        _dummy = Data(
                            x=torch.zeros(1, GCN_NODE_FEATURES, dtype=torch.float32),
                            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                            edge_attr=torch.zeros(1, 1, dtype=torch.float32),
                            num_nodes=1,
                        )
                        pyg_train         = [_dummy] * (val_start_idx  - train_start_idx)
                        pyg_val           = [_dummy] * (test_start_idx - val_start_idx)
                        pyg_seed_graphs   = [_dummy] * seq_length
                        pyg_future_graphs = [_dummy] * forecast_horizon
                    else:
                        # ── 1. Build per-window NX graphs ────────────────────
                        distance_metrics = ['euclidean', 'manhattan', 'hamming', 'amplitude_offset',
                                            'slope_consistency', 'phase_invariance', 'dtw', 'cid',
                                            'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
                        current_df_wide = df_wide_scaled if metric in distance_metrics else df_wide_global
                        compute_func = (compute_distances_1vsAll if metric_type == 'distance'
                                        else compute_similarities_1vsAll)

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

                        # ── 2. NX -> per-window PyG, align to timeline (per-day) ──
                        pyg_windows = build_pyg_graphs_from_nx_windows(
                            nx_graphs, current_df_wide, product_id,
                            window_size=window_size, step_size=step_size,
                        )
                        T_global = current_df_wide.shape[1]
                        pyg_aligned_global = _align_pyg_windows_to_timeline(
                            pyg_windows, window_size=window_size,
                            step_size=step_size, T=T_global,
                        )
                        product_offset = T_global - len(df_p)
                        pyg_train = pyg_aligned_global[product_offset + train_start_idx:
                                                       product_offset + val_start_idx]
                        pyg_val   = pyg_aligned_global[product_offset + val_start_idx:
                                                       product_offset + test_start_idx]

                        # Inference seed: the L per-day graphs ending at the last validation day.
                        seed_start = product_offset + test_start_idx - seq_length
                        seed_end   = product_offset + test_start_idx
                        pyg_seed_graphs = pyg_aligned_global[seed_start:seed_end]

                        # Future graphs aligned to each forecast day t in [test_start, test_start+H)
                        fut_start = product_offset + test_start_idx
                        fut_end   = fut_start + forecast_horizon
                        pyg_future_graphs = pyg_aligned_global[fut_start:fut_end]

                    # ── 3. Datasets / loaders (PER-STEP) ─────────────────────
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

                    # ── 4. Model + optimiser ─────────────────────────────────
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)

                    in_channels     = pyg_train[0].x.shape[1]    # 8 from _window_node_features
                    lstm_input_size = 1 + (len(EXOG_COLS) if EXOG_COLS else 0)
                    model = SimpleGCNLSTMForecaster(
                        in_channels=in_channels,
                        gcn_hidden=HIDDEN_SIZE,
                        d_g=D_G,
                        lstm_input_size=lstm_input_size,
                        lstm_hidden=HIDDEN_SIZE,
                        lstm_layers=NUM_LAYERS,
                        horizon=1,
                        dropout=DROPOUT,
                    ).to(device)
                    model.ablate_z = ablate_z
                    criterion  = nn.MSELoss()
                    criterion2 = nn.MSELoss()
                    optimizer  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
                    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=PATIENCE // 3,
                    )

                    # ── 5. Checkpoint paths (PER-STEP tag) ───────────────────
                    model_dir_label = (f"th{current_threshold}" if is_threshold_mode
                                       else f"pct{current_percentile}")
                    best_models_dir = os.path.join(
                        best_models_seed_dir, str(window_size), str(step_size),
                        metric, model_dir_label,
                    )
                    os.makedirs(best_models_dir, exist_ok=True)

                    prefix_star = "" if enable_edges else "star_"
                    if enable_second_degree:
                        prefix_star = "2nddegree_" + prefix_star
                    res_tag = f"_res_{MODEL_TYPE}" if USE_RESIDUALS else ""
                    param_label = (f"th_{current_threshold}" if is_threshold_mode
                                   else f"pct_{current_percentile}")
                    base_name = (
                        f"best_gcnlstm_perstep_{prefix_star}{product_id}_{metric}"
                        f"_w{window_size}_s{step_size}_{param_label}{res_tag}_seed_{seed}"
                    )
                    best_model_path = os.path.join(best_models_dir, f"{base_name}.pth")
                    history_path    = os.path.join(best_models_dir, f"{base_name}_history.pkl")
                    print(f"Resolved checkpoint: {best_model_path}")

                    # ── 6. Train (or reload) ─────────────────────────────────
                    if os.path.exists(best_model_path) and os.path.exists(history_path):
                        print(f"Loading existing model from {best_model_path}...")
                        model.load_state_dict(torch.load(best_model_path, map_location=device))
                        with open(history_path, 'rb') as f:
                            history = pickle.load(f)
                            train_losses = history['train_losses']
                            val_losses   = history['val_losses']
                    else:
                        print("Training new per-step GCN+LSTM model...")
                        diag_suffix = "_ablation" if ablate_z else ""
                        # Per-product diagnostics folder:
                        # diagnostics/<product_id>/diagnostics[_ablation].csv
                        product_diag_dir = os.path.join(
                            SCRIPT_DIR, "diagnostics", str(product_id),
                        )
                        os.makedirs(product_diag_dir, exist_ok=True)
                        diag_csv_path = os.path.join(
                            product_diag_dir,
                            DIAG_CSV_NAME.replace(".csv", f"{diag_suffix}.csv"),
                        )
                        # Mean number of edges per training window — characterises
                        # how "dense" the graph is at the current threshold.
                        # edge_index has shape (2, num_edges); each undirected
                        # edge appears twice, so we divide by 2.
                        num_edges_per_window = [
                            int(g.edge_index.shape[1] // 2) for g in pyg_train
                        ]
                        num_edges_mean = (
                            float(np.mean(num_edges_per_window))
                            if num_edges_per_window else 0.0
                        )
                        diag_meta = {
                            "product_id": product_id,
                            "store_id": store_id,
                            "metric": metric,
                            "window_size": window_size,
                            "step_size": step_size,
                            "threshold": current_threshold,
                            "percentile": current_percentile,
                            "enable_edges": enable_edges,
                            "enable_second_degree": enable_second_degree,
                            "num_edges_mean": num_edges_mean,
                        }
                        model, train_losses, val_losses, best_epoch, train_time = train_model(
                            seed=seed, epochs=EPOCHS, model=model,
                            train_loader=train_loader, val_loader=val_loader,
                            criterion=criterion, criterion2=criterion2,
                            optimizer=optimizer, device=device,
                            best_model_path=best_model_path if SAVE_MODELS else None,
                            scheduler=scheduler, patience=PATIENCE,
                            diag_csv_path=diag_csv_path, diag_meta=diag_meta,
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

                    # ── 7. Recursive inference (PER-STEP) ────────────────────
                    inf_threshold = fixed_threshold
                    print("Running Inference...")

                    # LSTM seed: last L target values of val, exog rows aligned so last row = exog_test[0]
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

                    # Build position->k map for the lag columns present in EXOG_COLS
                    lag_col_indices = {
                        EXOG_COLS.index(name): k
                        for name, k in LAG_K_BY_NAME.items()
                        if name in EXOG_COLS
                    }
                    # Build position->W map for rolling_mean_excl_W columns
                    rolling_mean_excl_col_indices = {}
                    for i, name in enumerate(EXOG_COLS):
                        if name.startswith(ROLLING_MEAN_EXCL_PREFIX):
                            try:
                                W = int(name[len(ROLLING_MEAN_EXCL_PREFIX):])
                            except ValueError:
                                continue
                            rolling_mean_excl_col_indices[i] = W
                    # Raw history that ends at the day BEFORE the first forecast step
                    target_history_unscaled = np.concatenate([train, val]).astype(np.float32)

                    forecast = _recursive_forecast_gcn_perstep(
                        model=model,
                        ts_seed=ts_seed,
                        initial_graphs=pyg_seed_graphs,
                        future_graphs=pyg_future_graphs,
                        exog_test_scaled=exog_test_scaled if EXOG_COLS else None,
                        scaler=scaler,
                        horizon=forecast_horizon,
                        device=device,
                        target_history_unscaled=target_history_unscaled if EXOG_COLS else None,
                        lag_col_indices=lag_col_indices if EXOG_COLS else None,
                        rolling_mean_excl_col_indices=rolling_mean_excl_col_indices if EXOG_COLS else None,
                        exog_scaler=exog_scaler if EXOG_COLS else None,
                    )

                    valid_mask     = ~np.isnan(forecast)
                    valid_test     = test[valid_mask]
                    valid_forecast = np.array(forecast)[valid_mask]

                    rmse = mae = bias = score = pocid = None
                    try:
                        rmse, mae, bias, score, pocid = compute_metrics(valid_test, valid_forecast)
                    except Exception:
                        if len(valid_test) > 0:
                            rmse  = float(np.sqrt(mean_squared_error(valid_test, valid_forecast)))
                            mae   = float(mean_absolute_error(valid_test, valid_forecast))
                            bias  = float(np.mean(valid_forecast - valid_test))
                            score = float(r2_score(valid_test, valid_forecast))

                    th_str = f"{inf_threshold:.4f}" if inf_threshold is not None else "N/A"
                    param_str_label = (f"th:{current_threshold}" if is_threshold_mode
                                       else f"pct:{current_percentile} (val:{th_str})")
                    az_str = "ablation" if ablate_z else "full"
                    label_name = (f"{param_str_label}|w:{window_size}|st:{step_size}"
                                  f"|e:{enable_edges}|2nd:{enable_second_degree}|az:{az_str}")

                    results_by_w_s[key]['forecasts'][label_name]    = forecast
                    results_by_w_s[key]['train_losses'][label_name] = train_losses
                    results_by_w_s[key]['val_losses'][label_name]   = val_losses
                    results_by_w_s[key]['rmse'][label_name]         = rmse
                    results_by_w_s[key]['mae'][label_name]          = mae
                    results_by_w_s[key]['bias'][label_name]         = bias
                    results_by_w_s[key]['score'][label_name]        = score
                    results_by_w_s[key]['pocid'][label_name]        = pocid

                    print(f"Finished {metric} @ {param_val} -> RMSE: {rmse}\n")

                    # ── Append to persistent CSV (per metric) ────────────────
                    import csv
                    csv_results_path = os.path.join(SCRIPT_DIR, f"{metric}.csv")
                    file_exists = os.path.exists(csv_results_path)
                    with open(csv_results_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow([
                                "product_id", "store_id", "seed", "metric",
                                "window_size", "step_size", "threshold", "percentile",
                                "enable_edges", "enable_second_degree", "ablate_z",
                                "rmse", "mae", "bias", "r2_score", "pocid",
                            ])
                        writer.writerow([
                            product_id, store_id, seed, metric,
                            window_size, step_size,
                            current_threshold if current_threshold is not None else "",
                            current_percentile if current_percentile is not None else "",
                            enable_edges, enable_second_degree, ablate_z,
                            rmse, mae, bias, score, pocid,
                        ])

                # ── Per-metric combined plot (all thresholds) ────────────────
                train_index = df_p[DATE_COL][train_slice].values
                val_index   = df_p[DATE_COL][val_slice].values
                test_index  = df_p[DATE_COL][test_slice].values

                grouped_results = {}
                for (az, p, w, s), res_dicts in results_by_w_s.items():
                    key = (az, w, s)
                    if key not in grouped_results:
                        grouped_results[key] = {
                            'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                            'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {},
                        }
                    for k in grouped_results[key]:
                        grouped_results[key][k].update(res_dicts[k])

                import hashlib
                for (az, w, s), res_dicts in grouped_results.items():
                    raw_str = ("_".join(map(str, thresholds))
                               if thresholds is not None and len(thresholds) > 0 and percentiles is None
                               else "_".join(map(str, percentiles)))
                    values_str = hashlib.md5(raw_str.encode()).hexdigest()[:8]

                    sub_dir = os.path.join(
                        grid_search_plots_dir, metric_type, f'window_{w}', f'step_{s}',
                        f'item_{product_id}', values_str,
                    )
                    os.makedirs(sub_dir, exist_ok=True)
                    ablation_suffix = "_ablation" if az else ""
                    save_plot_path = os.path.join(
                        sub_dir,
                        f"item_{product_id}_{metric}_seed_{seed}_all_configs{ablation_suffix}.html",
                    )
                    emb_title = (
                        f'GCN+LSTM Ablation (z=0) ({metric} | Seed={seed} | W={w} | S={s})'
                        if az else
                        f'GCN+LSTM (per-step) Forecasts ({metric} | Seed={seed} | W={w} | S={s})'
                    )

                    if SAVE_PLOTS:
                        print(f"Saving combined plot to: {os.path.abspath(save_plot_path)}")
                        plot_results(
                            train, val, test, res_dicts['forecasts'],
                            train_index, val_index, test_index,
                            res_dicts['train_losses'], res_dicts['val_losses'],
                            metric=metric, embedding_strategy='gcn_perstep',
                            window_size=w, step_size=s, threshold=None, percentile=None,
                            target_col=TARGET_COL,
                            title=f'{emb_title} (Item={product_id})',
                            seed=seed, save_path=save_plot_path,
                            rmse=res_dicts['rmse'], mae=res_dicts['mae'],
                            bias=res_dicts['bias'], score=res_dicts['score'],
                            pocid=res_dicts['pocid'],
                        )

                # ── Merged overlay: GCN+LSTM vs Ablation on the same plot ────
                if SAVE_PLOTS:
                    ws_combos = set((w, s) for (az, w, s) in grouped_results.keys())
                    for (w, s) in ws_combos:
                        full_key = (False, w, s)
                        abl_key  = (True,  w, s)
                        if full_key not in grouped_results or abl_key not in grouped_results:
                            continue

                        raw_str = ("_".join(map(str, thresholds))
                                   if thresholds is not None and len(thresholds) > 0 and percentiles is None
                                   else "_".join(map(str, percentiles)))
                        values_str = hashlib.md5(raw_str.encode()).hexdigest()[:8]
                        sub_dir = os.path.join(
                            grid_search_plots_dir, metric_type, f'window_{w}', f'step_{s}',
                            f'item_{product_id}', values_str,
                        )
                        os.makedirs(sub_dir, exist_ok=True)

                        # Merge forecasts/losses/metrics from both sides, prefixing labels
                        merged = {k: {} for k in ('forecasts', 'train_losses', 'val_losses',
                                                   'rmse', 'mae', 'bias', 'score', 'pocid')}
                        for prefix, src_key in [("GCN", full_key), ("Ablation", abl_key)]:
                            src = grouped_results[src_key]
                            for lbl in src['forecasts']:
                                new_lbl = f"{prefix}/{lbl}"
                                for k in merged:
                                    merged[k][new_lbl] = src[k].get(lbl)

                        merged_path = os.path.join(
                            sub_dir,
                            f"item_{product_id}_{metric}_seed_{seed}_all_configs_merged.html",
                        )
                        print(f"Saving merged overlay plot to: {os.path.abspath(merged_path)}")
                        plot_results(
                            train, val, test, merged['forecasts'],
                            train_index, val_index, test_index,
                            merged['train_losses'], merged['val_losses'],
                            metric=metric, embedding_strategy='gcn_perstep',
                            window_size=w, step_size=s, threshold=None, percentile=None,
                            target_col=TARGET_COL,
                            title=(f'GCN+LSTM vs Ablation (z=0) — '
                                   f'{metric} | Seed={seed} | W={w} | S={s} | Item={product_id}'),
                            seed=seed, save_path=merged_path,
                            rmse=merged['rmse'], mae=merged['mae'],
                            bias=merged['bias'], score=merged['score'],
                            pocid=merged['pocid'],
                        )
            # ── Per-seed timing (this product) ─────────────────────────────────────
            seed_elapsed = time.time() - seed_t0
            seed_times[seed] = seed_times.get(seed, 0.0) + seed_elapsed
            product_seed_times.append((product_id, store_id, seed, seed_elapsed))
            print(f"\n[TIMING] Product {product_id} | seed {seed}: {seed_elapsed:.1f} s "
                  f"({seed_elapsed/60:.2f} min)")

    # ── Total timing & persist ────────────────────────────────────────────────
    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    for s, t in sorted(seed_times.items()):
        print(f"  Seed {s}: total {t:.1f} s  ({t/60:.2f} min)  across {len(PRODUCTS_TO_TEST)} products")
    print(f"  TOTAL    : {total_elapsed:.1f} s  ({total_elapsed/60:.2f} min)")

    with open(timings_csv_path, "w", newline="") as fh:
        w = _csv_mod.writer(fh)
        w.writerow(["product_id", "store_id", "seed", "seconds"])
        for pid, sid, sd, sec in product_seed_times:
            w.writerow([pid, sid, sd, f"{sec:.3f}"])
        for sd, sec in sorted(seed_times.items()):
            w.writerow(["TOTAL_SEED", "", sd, f"{sec:.3f}"])
        w.writerow(["TOTAL_ALL", "", "", f"{total_elapsed:.3f}"])
    print(f"  Timings written to: {timings_csv_path}")
    # ── Correlation plots from accumulated CSVs ───────────────────────────
    if SAVE_PLOTS:
        import matplotlib.pyplot as plt
        print("\nGenerating Correlation Plots across all collected CSVs...")
        for csv_file in os.listdir(SCRIPT_DIR):
            if csv_file.endswith('.csv') and csv_file != 'no_emb.csv':
                metric_name = csv_file.replace('.csv', '')
                csv_path = os.path.join(SCRIPT_DIR, csv_file)
                try:
                    res_df = pd.read_csv(csv_path)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    fig.suptitle(f'Threshold vs RMSE and MAE | Metric: {metric_name}', fontsize=16)

                    x_col = 'threshold' if res_df['threshold'].notna().any() else 'percentile'
                    plot_data = res_df.dropna(subset=[x_col, 'rmse', 'mae']).sort_values(by=x_col)
                    if plot_data.empty:
                        continue

                    ax1.plot(plot_data[x_col], plot_data['rmse'], marker='o', linestyle='-', color='b')
                    ax1.set_title(f'{x_col.capitalize()} vs RMSE'); ax1.set_xlabel(x_col.capitalize())
                    ax1.set_ylabel('RMSE'); ax1.grid(True)

                    ax2.plot(plot_data[x_col], plot_data['mae'], marker='s', linestyle='-', color='r')
                    ax2.set_title(f'{x_col.capitalize()} vs MAE'); ax2.set_xlabel(x_col.capitalize())
                    ax2.set_ylabel('MAE'); ax2.grid(True)

                    plot_save_path = os.path.join(SCRIPT_DIR, f"{metric_name}_correlation_plot.png")
                    plt.tight_layout()
                    plt.savefig(plot_save_path)
                    plt.close()
                    print(f"Saved correlation plot for {metric_name} at {plot_save_path}")
                except Exception as e:
                    print(f"Failed to generate plot for {csv_file}: {e}")


if __name__ == '__main__':
    main()
