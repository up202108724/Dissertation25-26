"""
Wall-clock timing benchmark.

Measures total time (train + inference) for each of the six methods on
1, 10, 100 and 1000 products using a single seed.  Results are written to
timing_benchmark.csv and plotted in timing_benchmark.html.

Reduce BENCHMARK_EPOCHS / BENCHMARK_PATIENCE for a quick smoke-test.
"""

import os
import sys
import time
import csv
import random
import itertools
import threading

import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

from utils import (
    generate_exogenous_features, neighbourhood_graph,
    compute_similarities_1vsAll, compute_distances_1vsAll,
    _save_node_features_csv,
)
from node_feature_builders import node_feature_width as _nf_width, resolve_feature_names
from gcn_lstm_dataset import (
    GCN_LSTMTimeSeriesDataset,
    collate_pyg_ts as collate_pyg_ts_lstm,
    build_pyg_graphs_from_nx_windows as build_pyg_graphs_lstm,
)
from gat_tdmlp_dataset import (
    GCNMLPTimeSeriesDataset,
    collate_pyg_ts as collate_pyg_ts_mlp,
    build_pyg_graphs_from_nx_windows as build_pyg_graphs_mlp,
)
from gcn_models import (
    SimpleGCNLSTMForecaster, SimpleGCNMLPForecaster,
    SimpleGATLSTMForecaster, SimpleGATMLPForecaster,
)
from inference_strategies import (
    _recursive_forecast_gcn_lstm_perstep,
    _recursive_forecast_gcn_mlp_perstep,
)
from train_methods import train_gcn_lstm_model, train_gcn_mlpmodel

from LSTMBaseline.dataset import TimeSeriesDataset as LSTMBaselineDataset
from LSTMBaseline.lstm import LSTM as LSTMBaseline
from LSTMBaseline.lstm_train import train_model as train_lstm_baseline
from LSTMBaseline.lstm_inference import recursive_inference as recursive_forecast_lstm_baseline

from MLP_Baseline.train import train_mlp_forecaster, TrainConfig
from MLP_Baseline.inference import recursive_inference_dynamic_exog
from MLP_Baseline.utils import ExogenousScaler

# ── Benchmark config ───────────────────────────────────────────────────────
SEED            = 42
PRODUCT_COUNTS  = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
METHODS         = ['lstm_baseline', 'mlp_baseline', 'gcn_lstm', 'gcn_mlp', 'gat_lstm', 'gat_mlp']

# Reduce these for a quick smoke-test; increase to match your real training run.
BENCHMARK_EPOCHS   = 1000
BENCHMARK_PATIENCE = 100

# ── Dataset paths ──────────────────────────────────────────────────────────
DATA_PATH     = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/data_andre_classified.feather'))
TOP_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/top_12500.feather'))
DATE_COL      = 'date'
TARGET_COL    = 'value'

# ── Shared hyperparams (mirror the main runner) ────────────────────────────
val_size         = 31
forecast_horizon = 153
train_size       = 761 - val_size - forecast_horizon
lookback_window  = 30
BATCH_SIZE       = 32
LEARNING_RATE    = 0.001
HIDDEN_SIZE      = 32
NUM_LAYERS       = 1
DROPOUT          = 0.0
D_G              = 16
ATTENTION_HEADS  = 4
MLP_HIDDEN_SIZES = (128, 64)

# GNN graph config (single config for the benchmark)
GRAPH_METRIC    = 'spearman'
GRAPH_THRESHOLD = 0.634
WINDOW_SIZE     = 30
STEP_SIZE       = 1
NODE_FEAT_MODE  = 'catch24_minmaxlast'

EXOG_COLS_LSTM = [
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
EXOG_COLS_MLP = [
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "dom_sin", "dom_cos", "wom_sin", "wom_cos",
    "month_sin", "month_cos", "quarter_sin", "quarter_cos", "woy_sin", "woy_cos",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "rolling_mean_excl_7",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_bridge_day",
]
EXOG_COLS = list(dict.fromkeys(EXOG_COLS_LSTM + EXOG_COLS_MLP))

LAG_K_BY_NAME         = {"lag_1": 1, "lag_7": 7, "lag_30": 30}
ROLLING_MEAN_EXCL_PFX = "rolling_mean_excl_"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEM_SAMPLE_INTERVAL = 0.5   # seconds between memory samples


class MemorySampler:
    """Background thread that records (elapsed_s, gpu_mb, cpu_mb) at a fixed interval."""

    def __init__(self, interval: float = MEM_SAMPLE_INTERVAL):
        self.interval = interval
        self.samples: list = []   # [(elapsed_s, gpu_mb, cpu_mb), ...]
        self._stop = threading.Event()
        self._t0: float = 0.0
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process()

    def start(self):
        self.samples = []
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self):
        while not self._stop.wait(self.interval):
            elapsed = time.time() - self._t0
            gpu_mb  = (torch.cuda.memory_allocated() / 1e6
                       if torch.cuda.is_available() else 0.0)
            cpu_mb  = self._proc.memory_info().rss / 1e6
            self.samples.append((elapsed, gpu_mb, cpu_mb))


def _gpu_peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def _reset_gpu_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ── Helpers ────────────────────────────────────────────────────────────────
def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _node_feature_width(mode, window_size):
    names = resolve_feature_names(mode, extra_stats=True)
    return _nf_width(names, window_size=window_size)


def _make_pad_graph(template: Data) -> Data:
    in_feats = template.x.shape[1]
    return Data(
        x=torch.zeros(1, in_feats, dtype=torch.float32),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        edge_attr=torch.zeros(1, 1, dtype=torch.float32),
        num_nodes=1,
    )


def _align_pyg_windows(pyg_windows, window_size, T):
    pad = _make_pad_graph(pyg_windows[0])
    aligned = [pad] * window_size + list(pyg_windows)
    if len(aligned) < T:
        aligned += [aligned[-1]] * (T - len(aligned))
    return aligned[:T]


def _build_gnn_graphs(product_id, df_wide, metric_type, cat_labels, global_val_start_idx,
                      product_offset, train_start_idx, val_start_idx, test_start_idx,
                      build_graphs_fn):
    """Build and slice per-day PyG graphs for one product.
    Returns (pyg_train, pyg_val, pyg_seed, pyg_future, threshold, graph_build_seconds).
    """
    _gb_t0 = time.time()
    compute_func = (compute_distances_1vsAll if metric_type == 'distance'
                    else compute_similarities_1vsAll)
    nx_graphs, fixed_threshold = neighbourhood_graph(
        product_id=product_id, df=df_wide,
        metric=GRAPH_METRIC, metric_type=metric_type,
        window_size=WINDOW_SIZE, compute_func=compute_func,
        threshold=GRAPH_THRESHOLD, percentile=None,
        step_size=STEP_SIZE, cat_labels=cat_labels, plot_dir=None,
        residuals=False, enable_edges_within_star=True, enable_second_degree=False,
        train_end_idx=global_val_start_idx,
    )
    T_global = df_wide.shape[1]
    pyg_windows = build_graphs_fn(
        nx_graphs, df_wide, product_id,
        window_size=WINDOW_SIZE, step_size=STEP_SIZE,
        node_feature_mode=NODE_FEAT_MODE,
    )
    aligned = _align_pyg_windows(pyg_windows, WINDOW_SIZE, T_global)
    pyg_train  = aligned[product_offset + train_start_idx: product_offset + val_start_idx]
    pyg_val    = aligned[product_offset + val_start_idx:   product_offset + test_start_idx]
    seed_start = product_offset + test_start_idx - lookback_window
    pyg_seed   = aligned[seed_start: product_offset + test_start_idx]
    pyg_future = aligned[product_offset + test_start_idx: product_offset + test_start_idx + forecast_horizon]
    graph_build_s = time.time() - _gb_t0
    return pyg_train, pyg_val, pyg_seed, pyg_future, fixed_threshold, graph_build_s


def _ts_seed_and_lag_maps(val_scaled, exog_val_scaled, exog_test_scaled, exog_cols):
    if exog_cols:
        exog_seed_rows = np.vstack([
            exog_val_scaled[-(lookback_window - 1):],
            exog_test_scaled[0:1],
        ])
        ts_seed = np.column_stack([
            val_scaled[-lookback_window:].reshape(-1, 1),
            exog_seed_rows,
        ]).astype(np.float32)
    else:
        ts_seed = val_scaled[-lookback_window:].reshape(-1, 1).astype(np.float32)

    lag_col_indices = {
        exog_cols.index(n): k for n, k in LAG_K_BY_NAME.items() if n in exog_cols
    } if exog_cols else {}
    rolling_col_indices = {}
    if exog_cols:
        for i, n in enumerate(exog_cols):
            if n.startswith(ROLLING_MEAN_EXCL_PFX):
                try:
                    rolling_col_indices[i] = int(n[len(ROLLING_MEAN_EXCL_PFX):])
                except ValueError:
                    pass
    return ts_seed, lag_col_indices, rolling_col_indices


# ── Per-method runners (return elapsed seconds for one product) ────────────

def _run_lstm_baseline(product_id, store_id, df_p, train, val, test,
                       train_scaled, val_scaled, scaler,
                       exog_train_scaled, exog_val_scaled, exog_test_scaled, exog_scaler):
    lstm_input = 1 + (len(EXOG_COLS_LSTM) if EXOG_COLS_LSTM else 0)
    model = LSTMBaseline(
        input_size=lstm_input, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, dropout=DROPOUT,
    ).to(device)
    train_ds = LSTMBaselineDataset(train_scaled, exog_train_scaled, lookback_window)
    val_ds   = LSTMBaselineDataset(val_scaled,   exog_val_scaled,   lookback_window)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=BENCHMARK_PATIENCE // 3,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    model, _, _, _, _ = train_lstm_baseline(
        epochs=BENCHMARK_EPOCHS, model=model,
        train_loader=train_loader, val_loader=val_loader,
        exog_cols=EXOG_COLS_LSTM,
        criterion=nn.MSELoss(), criterion2=nn.MSELoss(),
        optimizer=optimizer, device=device,
        best_model_path=None, scheduler=scheduler, patience=BENCHMARK_PATIENCE,
    )
    exog_test_raw = df_p[EXOG_COLS_LSTM].values[-forecast_horizon:] if EXOG_COLS_LSTM else None
    test_start_idx = len(df_p) - forecast_horizon
    recursive_forecast_lstm_baseline(
        model=model, test_start_idx=test_start_idx, seq_length=lookback_window,
        val_scaled=val_scaled,
        exog_val_scaled=exog_val_scaled, exog_test_scaled=exog_test_scaled,
        exog_test=exog_test_raw, scaler=scaler, exog_scaler=exog_scaler,
        df_product=df_p, device=device, exog_cols=EXOG_COLS_LSTM,
        forecast_window=forecast_horizon, strategy='best_val',
        item_id=product_id, store_id=store_id, loss_type='MSELoss',
        script_dir=SCRIPT_DIR,
    )
    return 0.0   # graph_build_seconds (no graphs for this method)


def _run_mlp_baseline(product_id, store_id, df_p, train, val, test,
                      train_slice, val_slice, test_slice, scaler, seed):
    mlp_exog_scaler = ExogenousScaler(continuous_strategy='minmax')
    mlp_exog_scaler.fit(df_p[EXOG_COLS_MLP].iloc[train_slice], EXOG_COLS_MLP)
    cfg = TrainConfig(
        lookback=lookback_window, horizon=1, batch_size=BATCH_SIZE,
        train_size=train_size, val_size=val_size, lr=LEARNING_RATE,
        dropout=DROPOUT, epochs=BENCHMARK_EPOCHS, patience=BENCHMARK_PATIENCE,
        hidden_sizes=MLP_HIDDEN_SIZES, device=str(device),
    )
    mlp_model, _, _, _, _ = train_mlp_forecaster(
        df=df_p, cfg=cfg, seed=seed, loss_type='mse',
        product_id=f"{product_id}_{store_id}", scaler=scaler, target_channel=0,
        target_col=TARGET_COL, exog_cols=EXOG_COLS_MLP, test_size=forecast_horizon,
        exog_scaler=mlp_exog_scaler,
    )
    hist_target = np.concatenate([train, val]).astype(np.float32)
    hist_exog   = df_p[EXOG_COLS_MLP].iloc[val_slice].iloc[-lookback_window:].reset_index(drop=True)
    future_exog = df_p[EXOG_COLS_MLP].iloc[test_slice].reset_index(drop=True)
    recursive_inference_dynamic_exog(
        model=mlp_model, target_scaler=scaler, exog_scaler=mlp_exog_scaler,
        exog_cols=EXOG_COLS_MLP, history_target_unscaled=hist_target,
        history_exog_unscaled=hist_exog, future_exog_unscaled=future_exog,
        target_channel=0, device=str(device),
    )
    return 0.0   # graph_build_seconds (no graphs for this method)


def _run_gnn_variant(
    variant, product_id, store_id, df_p, df_wide, cat_labels,
    global_val_start_idx, product_offset,
    train, val, test, train_slice, val_slice, test_slice,
    train_start_idx, val_start_idx, test_start_idx,
    train_scaled, val_scaled, scaler,
    exog_train_by_head, exog_val_by_head, exog_test_by_head, exog_scalers_by_head,
    seed,
):
    head = 'mlp' if variant.endswith('mlp') else 'lstm'
    exog_cols        = EXOG_COLS_LSTM if head == 'lstm' else EXOG_COLS_MLP
    exog_train_sc    = exog_train_by_head[head]
    exog_val_sc      = exog_val_by_head[head]
    exog_test_sc     = exog_test_by_head[head]
    exog_scaler      = exog_scalers_by_head[head]

    if head == 'lstm':
        build_fn   = build_pyg_graphs_lstm
        DatasetCls = GCN_LSTMTimeSeriesDataset
        collate_fn = collate_pyg_ts_lstm
    else:
        build_fn   = build_pyg_graphs_mlp
        DatasetCls = GCNMLPTimeSeriesDataset
        collate_fn = collate_pyg_ts_mlp

    pyg_train, pyg_val, pyg_seed, pyg_future, _, graph_build_s = _build_gnn_graphs(
        product_id, df_wide, 'similarity', cat_labels, global_val_start_idx,
        product_offset, train_start_idx, val_start_idx, test_start_idx, build_fn,
    )

    use_pin = torch.cuda.is_available()
    train_ds = DatasetCls(train_scaled, exog_train_sc, lookback_window,
                          pyg_train, graph_window_size=WINDOW_SIZE)
    val_ds   = DatasetCls(val_scaled,   exog_val_sc,   lookback_window,
                          pyg_val,   graph_window_size=WINDOW_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=use_pin, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=use_pin, collate_fn=collate_fn)

    in_channels     = pyg_train[0].x.shape[1]
    lstm_input_size = 1 + (len(exog_cols) if exog_cols else 0)

    if variant == 'gcn_lstm':
        model = SimpleGCNLSTMForecaster(
            in_channels=in_channels, gcn_hidden=HIDDEN_SIZE, d_g=D_G,
            lstm_input_size=lstm_input_size, lstm_hidden=HIDDEN_SIZE,
            lstm_layers=NUM_LAYERS, horizon=1, dropout=DROPOUT,
        )
    elif variant == 'gat_lstm':
        model = SimpleGATLSTMForecaster(
            in_channels=in_channels, gat_hidden=HIDDEN_SIZE, d_g=D_G,
            lstm_input_size=lstm_input_size, lstm_hidden=HIDDEN_SIZE,
            lstm_layers=NUM_LAYERS, horizon=1, dropout=DROPOUT,
            attention_heads=ATTENTION_HEADS,
        )
    elif variant == 'gcn_mlp':
        model = SimpleGCNMLPForecaster(
            in_channels=in_channels, gcn_hidden=HIDDEN_SIZE, d_g=D_G,
            ts_input_size=lstm_input_size, lookback_window=lookback_window,
            hidden_sizes=MLP_HIDDEN_SIZES, horizon=1, dropout=DROPOUT,
        )
    else:  # gat_mlp
        model = SimpleGATMLPForecaster(
            in_channels=in_channels, gat_hidden=HIDDEN_SIZE, gat_heads=ATTENTION_HEADS, d_g=D_G,
            ts_input_size=lstm_input_size, lookback_window=lookback_window,
            hidden_sizes=MLP_HIDDEN_SIZES, horizon=1, dropout=DROPOUT,
        )
    model = model.to(device)
    model.ablate_z = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=BENCHMARK_PATIENCE // 3,
    )
    train_fn = train_gcn_lstm_model if head == 'lstm' else train_gcn_mlpmodel
    diag_dir = os.path.join(SCRIPT_DIR, 'benchmark_diag', str(product_id))
    os.makedirs(diag_dir, exist_ok=True)
    diag_csv = os.path.join(diag_dir, f'diag_{variant}_exp{seed}.csv')
    diag_meta = {'product_id': product_id, 'store_id': store_id, 'metric': GRAPH_METRIC,
                 'window_size': WINDOW_SIZE, 'step_size': STEP_SIZE,
                 'threshold': GRAPH_THRESHOLD, 'percentile': None,
                 'enable_edges': True, 'enable_second_degree': False, 'num_edges_mean': 0.0}
    model, _, _, _, _ = train_fn(
        seed=seed, epochs=BENCHMARK_EPOCHS, model=model,
        train_loader=train_loader, val_loader=val_loader,
        criterion=nn.MSELoss(), criterion2=nn.MSELoss(),
        optimizer=optimizer, device=device,
        best_model_path=None, scheduler=scheduler, patience=BENCHMARK_PATIENCE,
        diag_csv_path=diag_csv, diag_meta=diag_meta,
    )

    ts_seed, lag_col_idx, rolling_col_idx = _ts_seed_and_lag_maps(
        val_scaled, exog_val_sc, exog_test_sc, exog_cols,
    )
    target_history = np.concatenate([train, val]).astype(np.float32)

    if head == 'lstm':
        _recursive_forecast_gcn_lstm_perstep(
            model=model, ts_seed=ts_seed, initial_graphs=pyg_seed, future_graphs=pyg_future,
            exog_test_scaled=exog_test_sc, scaler=scaler, horizon=forecast_horizon,
            device=device, target_history_unscaled=target_history,
            lag_col_indices=lag_col_idx, rolling_mean_excl_col_indices=rolling_col_idx,
            exog_scaler=exog_scaler, initial_target_window=None, target_z_scaler=None,
            node_feature_mode=NODE_FEAT_MODE,
        )
    else:
        _recursive_forecast_gcn_mlp_perstep(
            model=model, ts_seed=ts_seed, initial_graphs=pyg_seed, future_graphs=pyg_future,
            exog_test_scaled=exog_test_sc, scaler=scaler, horizon=forecast_horizon,
            device=device, target_history_unscaled=target_history,
            lag_col_indices=lag_col_idx, rolling_mean_excl_col_indices=rolling_col_idx,
            exog_scaler=exog_scaler, exog_cols=exog_cols, node_feature_mode=NODE_FEAT_MODE,
        )
    return graph_build_s   # seconds spent building training graphs


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {device}")
    print(f"Seed: {SEED}  |  Product counts: {PRODUCT_COUNTS}  |  Methods: {METHODS}")
    print(f"Epochs: {BENCHMARK_EPOCHS}  |  Patience: {BENCHMARK_PATIENCE}\n")

    # ── Load & preprocess data (once) ─────────────────────────────────────
    print("Loading data...")
    df = pd.read_feather(DATA_PATH)
    if DATE_COL in df.index.names:
        df = df.reset_index()
    if df.index.name == DATE_COL:
        df = df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, 'item_id', 'store_id']).reset_index(drop=True)
    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS)
    full_df = df.copy()

    print("Loading top products...")
    top_df = pd.read_feather(TOP_DATA_PATH)
    all_products = (
        top_df[["item_id", "store_id"]].drop_duplicates()
        .sort_values(["item_id", "store_id"])
        .apply(lambda r: (int(r["item_id"]), int(r["store_id"])), axis=1)
        .tolist()
    )
    max_count = max(PRODUCT_COUNTS)
    if len(all_products) < max_count:
        print(f"Warning: only {len(all_products)} products available; "
              f"capping PRODUCT_COUNTS at {len(all_products)}.")
        global PRODUCT_COUNTS
        PRODUCT_COUNTS = [n for n in PRODUCT_COUNTS if n <= len(all_products)]
    products_pool = all_products[:max_count]

    cat_labels = (
        full_df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()
        if 'cat_label' in full_df.columns else {}
    )

    # Wide matrix for graph building (similarity → unscaled global)
    df_wide_global = full_df.pivot_table(
        index='item_id', columns=DATE_COL, values=TARGET_COL, aggfunc='sum'
    ).fillna(0)
    df_wide_global.columns = pd.to_datetime(df_wide_global.columns).strftime('%Y-%m-%d')

    Lcols = len(df_wide_global.columns)
    global_train_start_idx = max(0, Lcols - forecast_horizon - val_size - train_size)
    global_val_start_idx   = Lcols - forecast_horizon - val_size

    # ── Benchmark loop ────────────────────────────────────────────────────
    records      = []   # (n_products, method, total_s, s_per_prod, graph_build_s, peak_gpu_mb, peak_cpu_mb)
    mem_peaks    = []   # per-product: (n_products, method, product_idx, product_id, store_id, peak_gpu_mb, peak_cpu_mb)
    mem_timeline = []   # (n_products, method, elapsed_s, gpu_mb, cpu_mb)
    _proc        = psutil.Process()

    for n_products in PRODUCT_COUNTS:
        products = products_pool[:n_products]
        print(f"\n{'='*70}")
        print(f"n_products = {n_products}")
        print(f"{'='*70}")

        for method in METHODS:
            _set_seed(SEED)
            print(f"\n  [{method}]  timing {n_products} product(s)...")

            sampler = MemorySampler()
            sampler.start()
            t0 = time.time()
            graph_build_total = 0.0

            for prod_idx, (product_id, store_id) in enumerate(products):
                df_p = (
                    full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)]
                    .sort_values(DATE_COL).reset_index(drop=True)
                )
                required = forecast_horizon + val_size + train_size
                if len(df_p) < required:
                    print(f"    Skipping {product_id}/{store_id}: only {len(df_p)} rows.")
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

                scaler      = MinMaxScaler()
                train_sc    = scaler.fit_transform(train.reshape(-1, 1)).flatten()
                val_sc      = scaler.transform(val.reshape(-1, 1)).flatten()

                exog_train_by_head, exog_val_by_head = {}, {}
                exog_test_by_head, exog_scalers_by_head = {}, {}
                for _head, _cols in [('lstm', EXOG_COLS_LSTM), ('mlp', EXOG_COLS_MLP)]:
                    if _cols:
                        _sc = MinMaxScaler()
                        exog_train_by_head[_head] = _sc.fit_transform(df_p[_cols][train_slice].values)
                        exog_val_by_head[_head]   = _sc.transform(df_p[_cols][val_slice].values)
                        exog_test_by_head[_head]  = _sc.transform(df_p[_cols][test_slice].values)
                        exog_scalers_by_head[_head] = _sc
                    else:
                        exog_train_by_head[_head] = exog_val_by_head[_head] = exog_test_by_head[_head] = None
                        exog_scalers_by_head[_head] = None

                T_global       = df_wide_global.shape[1]
                product_offset = T_global - len(df_p)

                # ── Per-product memory tracking ───────────────────────────
                _reset_gpu_peak()
                cpu_before_mb = _proc.memory_info().rss / 1e6

                if method == 'lstm_baseline':
                    gb_s = _run_lstm_baseline(
                        product_id, store_id, df_p, train, val, test,
                        train_sc, val_sc, scaler,
                        exog_train_by_head['lstm'], exog_val_by_head['lstm'],
                        exog_test_by_head['lstm'], exog_scalers_by_head['lstm'],
                    )
                elif method == 'mlp_baseline':
                    gb_s = _run_mlp_baseline(
                        product_id, store_id, df_p, train, val, test,
                        train_slice, val_slice, test_slice, scaler, SEED,
                    )
                else:
                    gb_s = _run_gnn_variant(
                        variant=method,
                        product_id=product_id, store_id=store_id, df_p=df_p,
                        df_wide=df_wide_global, cat_labels=cat_labels,
                        global_val_start_idx=global_val_start_idx,
                        product_offset=product_offset,
                        train=train, val=val, test=test,
                        train_slice=train_slice, val_slice=val_slice, test_slice=test_slice,
                        train_start_idx=train_start_idx, val_start_idx=val_start_idx,
                        test_start_idx=test_start_idx,
                        train_scaled=train_sc, val_scaled=val_sc, scaler=scaler,
                        exog_train_by_head=exog_train_by_head,
                        exog_val_by_head=exog_val_by_head,
                        exog_test_by_head=exog_test_by_head,
                        exog_scalers_by_head=exog_scalers_by_head,
                        seed=SEED,
                    )

                graph_build_total += gb_s
                peak_gpu = _gpu_peak_mb()
                peak_cpu = max(_proc.memory_info().rss / 1e6, cpu_before_mb)
                mem_peaks.append((n_products, method, prod_idx, product_id, store_id,
                                  f'{peak_gpu:.2f}', f'{peak_cpu:.2f}'))

            elapsed = time.time() - t0
            sampler.stop()

            per_product = elapsed / n_products
            # overall peak across the sampler window
            all_gpu = [s[1] for s in sampler.samples] or [0.0]
            all_cpu = [s[2] for s in sampler.samples] or [0.0]
            peak_gpu_run = max(all_gpu)
            peak_cpu_run = max(all_cpu)
            records.append((n_products, method, elapsed, per_product,
                            graph_build_total, peak_gpu_run, peak_cpu_run))
            for elapsed_s, gpu_mb, cpu_mb in sampler.samples:
                mem_timeline.append((n_products, method, f'{elapsed_s:.3f}',
                                     f'{gpu_mb:.2f}', f'{cpu_mb:.2f}'))

            print(f"    -> {elapsed:.1f} s total  ({per_product:.1f} s/prod)"
                  f"  graph_build={graph_build_total:.1f} s"
                  f"  peak_gpu={peak_gpu_run:.0f} MB  peak_cpu={peak_cpu_run:.0f} MB")

    # ── Save CSVs ─────────────────────────────────────────────────────────
    csv_path = os.path.join(SCRIPT_DIR, 'timing_benchmark.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['n_products', 'method', 'total_seconds', 'seconds_per_product',
                    'graph_build_seconds', 'peak_gpu_mb', 'peak_cpu_mb'])
        for row in records:
            w.writerow([row[0], row[1], f'{row[2]:.3f}', f'{row[3]:.3f}',
                        f'{row[4]:.3f}', f'{row[5]:.2f}', f'{row[6]:.2f}'])
    print(f"\nTiming CSV:   {csv_path}")

    mem_peaks_path = os.path.join(SCRIPT_DIR, 'memory_peak.csv')
    with open(mem_peaks_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['n_products', 'method', 'product_idx', 'product_id', 'store_id',
                    'peak_gpu_mb', 'peak_cpu_mb'])
        for row in mem_peaks:
            w.writerow(row)
    print(f"Peak mem CSV: {mem_peaks_path}")

    mem_tl_path = os.path.join(SCRIPT_DIR, 'memory_timeline.csv')
    with open(mem_tl_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['n_products', 'method', 'elapsed_s', 'gpu_mb', 'cpu_mb'])
        for row in mem_timeline:
            w.writerow(row)
    print(f"Timeline CSV: {mem_tl_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(records, columns=['n_products', 'method', 'total_seconds', 'seconds_per_product'])

    method_colors = {
        'lstm_baseline': '#636EFA',
        'mlp_baseline':  '#EF553B',
        'gcn_lstm':      '#00CC96',
        'gcn_mlp':       '#AB63FA',
        'gat_lstm':      '#FFA15A',
        'gat_mlp':       '#19D3F3',
    }
    method_symbols = {
        'lstm_baseline': 'circle',
        'mlp_baseline':  'square',
        'gcn_lstm':      'diamond',
        'gcn_mlp':       'cross',
        'gat_lstm':      'triangle-up',
        'gat_mlp':       'star',
    }

    tick_vals = PRODUCT_COUNTS
    tick_text = [str(n) for n in PRODUCT_COUNTS]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Total time (train + inference)',
            'Time per product (scaling slope)',
        ),
        shared_xaxes=False,
        horizontal_spacing=0.10,
    )

    for method in METHODS:
        sub = df_res[df_res['method'] == method].sort_values('n_products')
        if sub.empty:
            continue
        color  = method_colors.get(method)
        symbol = method_symbols.get(method)
        shared_kwargs = dict(
            mode='lines+markers',
            name=method,
            line=dict(color=color, width=2),
            marker=dict(symbol=symbol, size=9, color=color),
            legendgroup=method,
        )

        # Left: total time
        fig.add_trace(go.Scatter(
            x=sub['n_products'], y=sub['total_seconds'],
            hovertemplate=(
                '<b>%{fullData.name}</b><br>'
                'Products: %{x}<br>'
                'Total: %{y:.1f} s<extra></extra>'
            ),
            **shared_kwargs,
        ), row=1, col=1)

        # Right: time per product — reveals whether scaling is linear or super-linear
        fig.add_trace(go.Scatter(
            x=sub['n_products'], y=sub['seconds_per_product'],
            hovertemplate=(
                '<b>%{fullData.name}</b><br>'
                'Products: %{x}<br>'
                'Per product: %{y:.2f} s<extra></extra>'
            ),
            showlegend=False,
            **{k: v for k, v in shared_kwargs.items() if k != 'showlegend'},
        ), row=1, col=2)

    axis_style = dict(type='log', tickvals=tick_vals, ticktext=tick_text,
                      title_text='Number of products', showgrid=True, gridcolor='#e8e8e8')
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(showgrid=True, gridcolor='#e8e8e8')
    fig.update_yaxes(title_text='Total time (s)',      type='log', row=1, col=1)
    fig.update_yaxes(title_text='Seconds per product', type='log', row=1, col=2)

    fig.update_layout(
        title=dict(
            text=(
                'Time complexity — all 6 methods<br>'
                f'<sup>Seed={SEED} | {BENCHMARK_EPOCHS} epochs | '
                f'{GRAPH_METRIC} @ {GRAPH_THRESHOLD}</sup>'
            ),
            x=0.5,
        ),
        legend=dict(
            title='Method', x=1.01, y=1.0,
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#cccccc', borderwidth=1,
        ),
        hovermode='x unified',
        template='plotly_white',
        width=1100, height=500,
    )

    html_path = os.path.join(SCRIPT_DIR, 'timing_benchmark.html')
    fig.write_html(html_path)
    print(f"Plot saved:  {html_path}")

    # Print summary table
    print("\n── Summary ─────────────────────────────────────────────────────────────────────")
    print(f"{'n_products':>12} {'method':>16} {'total_s':>10} {'s/prod':>8} "
          f"{'graph_s':>9} {'peak_gpu':>10} {'peak_cpu':>10}")
    print('-' * 79)
    for n, m, tot, per, gb, pgpu, pcpu in records:
        print(f"{n:>12} {m:>16} {tot:>10.1f} {per:>8.1f} "
              f"{gb:>9.1f} {pgpu:>9.0f}M {pcpu:>9.0f}M")


if __name__ == '__main__':
    main()
