"""
Sequential scalability benchmark — GCN / GAT / Graph2Vec assemble.

Runs 8 model variants sequentially over the same 500 products and records
cumulative TRAIN + INFERENCE time at N ∈ {1, 10, 29, 50, 100, 200, 500}.

Data loading mirrors graph2vec_gcn_gat_assmble_parallelized_time_elapsed.py:
  • full feature-engineered DataFrame from DATA_PATH
  • product list from TOP_DATA_PATH

Output: scalability_results.csv  +  printed summary table.
"""

from __future__ import annotations

import os
import sys
import csv
import time
import random
import tempfile
import traceback

import numpy as np
import pandas as pd

# ── Optional psutil for RAM tracking ─────────────────────────────────────────
try:
    import psutil as _psutil
    _PROC = _psutil.Process(os.getpid())
    def _proc_rss_mb() -> float:
        return _PROC.memory_info().rss / 1024 ** 2
except ImportError:
    def _proc_rss_mb() -> float:   # type: ignore[misc]
        return 0.0

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ── Paths & sys.path ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

from utils import (
    generate_exogenous_features, neighbourhood_graph,
    compute_distances_1vsAll, compute_similarities_1vsAll,
)

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
    SimpleGraph2VecLSTMForecaster, SimpleGraph2VecMLPForecaster,
)
from inference_strategies import (
    _recursive_forecast_gcn_lstm_perstep,
    _recursive_forecast_gcn_mlp_perstep,
    _align_pyg_windows_to_timeline,
)
from generate_graph2vecwithadaptativethreshold import load_or_generate_embeddings
from graph2vec_assemble import (
    recursive_forecast_graph2vec_perstep,
    embedding_to_single_node_data,
)
from train_methods import train_gcn_lstm_model, train_gcn_mlpmodel

from LSTMBaseline.dataset import TimeSeriesDataset as LSTMBaselineDataset
from LSTMBaseline.lstm import LSTM as LSTMBaseline
from LSTMBaseline.lstm_train import train_model as train_lstm_baseline
from LSTMBaseline.lstm_inference import recursive_inference as recursive_forecast_lstm_baseline

from MLP_Baseline.train import train_mlp_forecaster, TrainConfig
from MLP_Baseline.inference import recursive_inference_dynamic_exog
from MLP_Baseline.utils import ExogenousScaler

# ── Metric helpers ───────────────────────────────────────────────────────────
DISTANCE_METRICS = {
    'euclidean', 'hamming', 'amplitude_offset', 'slope_consistency',
    'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss',
    'manhattan', 'twed', 'erp', 'stid',
}
SIMILARITY_METRICS = {'pearson', 'spearman', 'kendall'}


def infer_metric_type(metric):
    if metric in DISTANCE_METRICS:
        return 'distance'
    if metric in SIMILARITY_METRICS:
        return 'similarity'
    raise ValueError(f"Unknown metric: {metric!r}")


# ── Dataset / run constants ───────────────────────────────────────────────────
DATA_PATH     = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/data_andre_classified.feather'))
TOP_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/top_12500.feather'))
DATE_COL   = 'date'
TARGET_COL = 'value'

val_size         = 61
forecast_horizon = 153
train_size       = 761 - val_size - forecast_horizon
lookback_window  = 30
BATCH_SIZE       = 32

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

grid_configs = [{'metric': 'spearman', 'thresholds': [0.634]}]
window_sizes              = [30]
step_sizes                = [1]
enable_edges_opts         = [True]
enable_second_degree_opts = [False]
node_feature_modes        = ['catch24_minmaxlast']

USE_RESIDUALS = False
MODEL_TYPE    = 'ridge'
EPOCHS        = 1000
PATIENCE      = 100

LEARNING_RATE_LSTM = 0.001
LEARNING_RATE_MLP  = 0.0001
HIDDEN_SIZE        = 32
NUM_LAYERS         = 1
DROPOUT_LSTM       = 0.0
DROPOUT_MLP        = 0.2
D_G                = 16
ATTENTION_HEADS    = 4
MLP_HIDDEN_SIZES   = (128, 64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LAG_K_BY_NAME            = {"lag_1": 1, "lag_7": 7, "lag_30": 30}
ROLLING_MEAN_EXCL_PREFIX = "rolling_mean_excl_"

# ── Benchmark configuration ───────────────────────────────────────────────────
PRODUCT_COUNTS = [1, 10, 29, 50, 100, 200, 500]
BENCH_SEED     = 42
BENCH_VARIANTS = [

    'gcn_lstm', 'gcn_mlp', 'gat_lstm', 'gat_mlp',
]

_CFG         = grid_configs[0]
METRIC       = _CFG['metric']
_THRESHOLDS  = _CFG.get('thresholds', [None])
IS_THRESHOLD = _THRESHOLDS not in (None, [None])
PARAM_VAL    = _THRESHOLDS[0] if IS_THRESHOLD else _CFG.get('percentiles', [None])[0]
WINDOW_SIZE  = window_sizes[0]
STEP_SIZE    = step_sizes[0]
ENABLE_EDGES = enable_edges_opts[0]
ENABLE_2ND   = enable_second_degree_opts[0]
NODE_FEAT_MODE = node_feature_modes[0]
RESULTS_CSV  = os.path.join(SCRIPT_DIR, "scalability_results.csv")

_DISTANCE_METRICS = [
    'euclidean', 'manhattan', 'hamming', 'amplitude_offset',
    'slope_consistency', 'phase_invariance', 'dtw', 'cid',
    'lorentzian', 'sbd', 'msm', 'edr', 'lcss',
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _variant_head(variant):
    return 'mlp' if variant.endswith('mlp') else 'lstm'


def build_forecaster(variant, in_channels, ts_input_size):
    head    = _variant_head(variant)
    dropout = DROPOUT_LSTM if head == 'lstm' else DROPOUT_MLP
    if variant == 'graph2vec_lstm':
        return SimpleGraph2VecLSTMForecaster(
            in_channels=in_channels, d_g=D_G,
            lstm_input_size=ts_input_size, lstm_hidden=HIDDEN_SIZE,
            lstm_layers=NUM_LAYERS, horizon=1, dropout=dropout,
        )
    if variant == 'graph2vec_mlp':
        return SimpleGraph2VecMLPForecaster(
            in_channels=in_channels, d_g=D_G,
            ts_input_size=ts_input_size, hidden_sizes=MLP_HIDDEN_SIZES,
            lookback_window=lookback_window, horizon=1, dropout=dropout,
        )
    if variant == 'gcn_lstm':
        return SimpleGCNLSTMForecaster(
            in_channels=in_channels, gcn_hidden=HIDDEN_SIZE, d_g=D_G,
            lstm_input_size=ts_input_size, lstm_hidden=HIDDEN_SIZE,
            lstm_layers=NUM_LAYERS, horizon=1, dropout=dropout,
        )
    if variant == 'gat_lstm':
        return SimpleGATLSTMForecaster(
            in_channels=in_channels, gat_hidden=HIDDEN_SIZE, d_g=D_G,
            lstm_input_size=ts_input_size, lstm_hidden=HIDDEN_SIZE,
            lstm_layers=NUM_LAYERS, horizon=1, dropout=dropout,
            attention_heads=ATTENTION_HEADS,
        )
    if variant == 'gcn_mlp':
        return SimpleGCNMLPForecaster(
            in_channels=in_channels, gcn_hidden=HIDDEN_SIZE, d_g=D_G,
            ts_input_size=ts_input_size, lookback_window=lookback_window,
            hidden_sizes=MLP_HIDDEN_SIZES, horizon=1, dropout=dropout,
        )
    if variant == 'gat_mlp':
        return SimpleGATMLPForecaster(
            in_channels=in_channels, gat_hidden=HIDDEN_SIZE,
            gat_heads=ATTENTION_HEADS, d_g=D_G,
            ts_input_size=ts_input_size, lookback_window=lookback_window,
            hidden_sizes=MLP_HIDDEN_SIZES, horizon=1, dropout=dropout,
        )
    raise ValueError(f"Unknown variant: {variant!r}")


# ── Shared (product-independent) data setup ───────────────────────────────────
def prepare_shared():
    print(f"Loading main dataset from {DATA_PATH} ...")
    df = pd.read_feather(DATA_PATH)
    if DATE_COL in df.index.names:
        df = df.reset_index() if DATE_COL not in df.columns else df.reset_index(drop=True)
    if df.index.name == DATE_COL:
        df = df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, 'item_id', 'store_id']).reset_index(drop=True)
    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS)
    full_df = df.copy()

    min_rows = forecast_horizon + val_size + train_size
    counts = full_df.groupby(['item_id', 'store_id']).size().reset_index(name='_cnt')
    products = (
        counts[counts['_cnt'] >= min_rows][['item_id', 'store_id']]
        .sort_values(['item_id', 'store_id'])
        .apply(lambda r: (int(r['item_id']), int(r['store_id'])), axis=1)
        .tolist()
    )
    print(f"  {len(products)} eligible (item_id, store_id) pairs (>= {min_rows} rows) found.")

    cat_labels_dict = (
        full_df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()
        if 'cat_label' in full_df.columns else {}
    )

    df_wide_global = full_df.pivot_table(
        index='item_id', columns=DATE_COL, values=TARGET_COL, aggfunc='sum'
    ).fillna(0)
    df_wide_global.columns = pd.to_datetime(df_wide_global.columns).strftime('%Y-%m-%d')

    Lcols = len(df_wide_global.columns)
    global_train_start_idx = max(0, Lcols - forecast_horizon - val_size - train_size)
    global_val_start_idx   = Lcols - forecast_horizon - val_size

    product_scalers = {}
    train_df_wide   = df_wide_global.iloc[:, global_train_start_idx:global_val_start_idx]
    df_wide_scaled  = df_wide_global.copy()
    for iid in df_wide_global.index:
        z = StandardScaler()
        z.fit(train_df_wide.loc[iid].values.reshape(-1, 1))
        product_scalers[iid] = z
        df_wide_scaled.loc[iid] = z.transform(
            df_wide_global.loc[iid].values.reshape(-1, 1)
        ).flatten()

    return dict(
        full_df=full_df, products=products,
        cat_labels_dict=cat_labels_dict,
        df_wide_global=df_wide_global, df_wide_scaled=df_wide_scaled,
        product_scalers=product_scalers,
        global_val_start_idx=global_val_start_idx,
    )


# ── Per-product slice ─────────────────────────────────────────────────────────
def prepare_product(shared, product_id, store_id):
    full_df = shared['full_df']
    df_p = (
        full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)]
        .sort_values(DATE_COL).reset_index(drop=True)
    )
    if len(df_p) < forecast_horizon + val_size + train_size:
        return None

    test_start_idx  = len(df_p) - forecast_horizon
    val_start_idx   = test_start_idx - val_size
    train_start_idx = val_start_idx - train_size

    train_slice = slice(train_start_idx, val_start_idx)
    val_slice   = slice(val_start_idx,   test_start_idx)
    test_slice  = slice(test_start_idx,  None)

    train = df_p[TARGET_COL][train_slice].values
    val   = df_p[TARGET_COL][val_slice].values
    test  = df_p[TARGET_COL][test_slice].values

    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
    val_scaled   = scaler.transform(val.reshape(-1, 1)).flatten()

    exog_by_head = {}
    for head, cols in [('lstm', EXOG_COLS_LSTM), ('mlp', EXOG_COLS_MLP)]:
        if cols:
            sc = MinMaxScaler()
            exog_by_head[head] = dict(
                scaler=sc,
                train=sc.fit_transform(df_p[cols][train_slice].values),
                val=sc.transform(df_p[cols][val_slice].values),
                test=sc.transform(df_p[cols][test_slice].values),
            )
        else:
            exog_by_head[head] = dict(scaler=None, train=None, val=None, test=None)

    return dict(
        df_p=df_p, train=train, val=val, test=test,
        scaler=scaler, train_scaled=train_scaled, val_scaled=val_scaled,
        train_start_idx=train_start_idx, val_start_idx=val_start_idx,
        test_start_idx=test_start_idx, test_slice=test_slice,
        exog_by_head=exog_by_head,
    )


# ── Graph / embedding pipeline ────────────────────────────────────────────────
def build_graphs_for_variant(variant, shared, prod, product_id):
    """
    Returns (pyg_train, pyg_val, pyg_seed_graphs, pyg_future_graphs,
             in_channels, graph2vec_model, current_df_wide).
    pyg_seed_graphs   — lookback_window graphs ending at test day 0.
    pyg_future_graphs — forecast_horizon graphs, one per forecast day.
    """
    metric_type     = infer_metric_type(METRIC)
    current_df_wide = (
        shared['df_wide_scaled'] if METRIC in _DISTANCE_METRICS
        else shared['df_wide_global']
    )
    T_global       = current_df_wide.shape[1]
    product_offset = T_global - len(prod['df_p'])

    if variant.startswith('graph2vec'):
        (graph_embeddings, g2v_model, _csv, _bt, _et, _thr) = load_or_generate_embeddings(
            product_id=product_id, metric=METRIC, metric_type=metric_type,
            window_size=WINDOW_SIZE, step_size=STEP_SIZE,
            threshold=PARAM_VAL if IS_THRESHOLD else None,
            percentile=None if IS_THRESHOLD else PARAM_VAL,
            dimensions=D_G, enable_edges_within_star=ENABLE_EDGES,
            enable_second_degree=ENABLE_2ND, use_residuals=USE_RESIDUALS,
            model_type=MODEL_TYPE, seed=BENCH_SEED, df=current_df_wide,
            cat_labels=shared['cat_labels_dict'],
            train_end_idx=shared['global_val_start_idx'], save_embeddings=False,
        )
        pyg_windows = [
            embedding_to_single_node_data(graph_embeddings[i], target_label=product_id)
            for i in range(len(graph_embeddings))
        ]
        graph2vec_model = g2v_model
    else:
        compute_func = (
            compute_distances_1vsAll if metric_type == 'distance'
            else compute_similarities_1vsAll
        )
        nx_graphs, _thr = neighbourhood_graph(
            product_id=product_id, df=current_df_wide, metric=METRIC,
            metric_type=metric_type, window_size=WINDOW_SIZE,
            compute_func=compute_func,
            threshold=PARAM_VAL if IS_THRESHOLD else None,
            percentile=None if IS_THRESHOLD else PARAM_VAL,
            step_size=STEP_SIZE, cat_labels=shared['cat_labels_dict'],
            plot_dir=None, residuals=USE_RESIDUALS,
            enable_edges_within_star=ENABLE_EDGES,
            enable_second_degree=ENABLE_2ND,
            train_end_idx=shared['global_val_start_idx'],
        )
        build_fn = (
            build_pyg_graphs_lstm if _variant_head(variant) == 'lstm'
            else build_pyg_graphs_mlp
        )
        pyg_windows = build_fn(
            nx_graphs, current_df_wide, product_id,
            window_size=WINDOW_SIZE, step_size=STEP_SIZE,
            node_feature_mode=NODE_FEAT_MODE,
        )
        graph2vec_model = None

    aligned = _align_pyg_windows_to_timeline(
        pyg_windows, window_size=WINDOW_SIZE, step_size=STEP_SIZE, T=T_global,
    )

    o         = product_offset
    tr_start  = prod['train_start_idx']
    val_start = prod['val_start_idx']
    tst_start = prod['test_start_idx']

    pyg_train         = aligned[o + tr_start  : o + val_start]
    pyg_val           = aligned[o + val_start : o + tst_start]
    pyg_seed_graphs   = aligned[o + tst_start - lookback_window : o + tst_start]
    pyg_future_graphs = aligned[o + tst_start : o + tst_start + forecast_horizon]

    in_channels = pyg_train[0].x.shape[1]
    return (
        pyg_train, pyg_val, pyg_seed_graphs, pyg_future_graphs,
        in_channels, graph2vec_model, current_df_wide,
    )


# ── Memory helpers ───────────────────────────────────────────────────────────
def _reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _read_peak_gpu_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


# ── One (variant, product): train + infer ────────────────────────────────────
def run_one(variant, shared, product_id, store_id):
    """Returns (build_s, train_s, infer_s, peak_gpu_mb, peak_ram_mb, epochs_trained, s_per_epoch)
    or None if the product is skipped.
    peak_gpu_mb   — peak GPU memory allocated during the whole run (0 on CPU-only).
    peak_ram_mb   — peak RSS increase during the whole run (requires psutil).
    epochs_trained — actual number of epochs run (early-stopping may cut short EPOCHS).
    s_per_epoch   — train_s / epochs_trained.
    """
    prod = prepare_product(shared, product_id, store_id)
    if prod is None:
        return None

    _reset_peak_memory()
    _ram_baseline_mb = _proc_rss_mb()

    # ── LSTM baseline ────────────────────────────────────────────────────────
    if variant == 'lstm_baseline':
        _seed_everything(BENCH_SEED)
        cols = EXOG_COLS_LSTM
        eb   = prod['exog_by_head']['lstm']
        model = LSTMBaseline(
            input_size=1 + (len(cols) if cols else 0),
            hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT_LSTM,
        ).to(device)
        tr_ds = LSTMBaselineDataset(
            prod['train_scaled'], eb['train'] if cols else None, lookback_window)
        va_ds = LSTMBaselineDataset(
            prod['val_scaled'],   eb['val']   if cols else None, lookback_window)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=False)
        va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_LSTM)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=PATIENCE // 3)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as _tf:
            tmp_path = _tf.name
        t_train0 = time.time()
        try:
            model, train_losses, _, _, _ = train_lstm_baseline(
                epochs=EPOCHS, model=model,
                train_loader=tr_ld, val_loader=va_ld,
                exog_cols=cols,
                criterion=nn.MSELoss(), criterion2=nn.MSELoss(),
                optimizer=opt, device=device,
                best_model_path=tmp_path, scheduler=sch, patience=PATIENCE,
            )
            if os.path.exists(tmp_path):
                model.load_state_dict(
                    torch.load(tmp_path, map_location=device, weights_only=True))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        train_s = time.time() - t_train0
        epochs_trained = len(train_losses)

        t_infer0 = time.time()
        recursive_forecast_lstm_baseline(
            model=model,
            test_start_idx=prod['test_start_idx'],
            seq_length=lookback_window,
            val_scaled=prod['val_scaled'],
            exog_val_scaled=eb['val']  if cols else None,
            exog_test_scaled=eb['test'] if cols else None,
            exog_test=prod['df_p'][cols][prod['test_slice']].values if cols else None,
            scaler=prod['scaler'],
            exog_scaler=eb['scaler'] if cols else None,
            df_product=prod['df_p'],
            device=device,
            exog_cols=cols,
            forecast_window=forecast_horizon,
            strategy='scalability',
            item_id=product_id,
            store_id=store_id,
            loss_type='MSELoss',
            script_dir=SCRIPT_DIR,
        )
        infer_s = time.time() - t_infer0
        peak_gpu_mb = _read_peak_gpu_mb()
        peak_ram_mb = max(0.0, _proc_rss_mb() - _ram_baseline_mb)
        s_per_epoch = train_s / epochs_trained if epochs_trained else 0.0
        return 0.0, train_s, infer_s, peak_gpu_mb, peak_ram_mb, epochs_trained, s_per_epoch

    # ── MLP baseline ─────────────────────────────────────────────────────────
    if variant == 'mlp_baseline':
        _seed_everything(BENCH_SEED)
        mlp_exog_scaler = ExogenousScaler(continuous_strategy='minmax')
        mlp_exog_scaler.fit(
            prod['df_p'][EXOG_COLS_MLP].iloc[
                slice(prod['train_start_idx'], prod['val_start_idx'])
            ],
            EXOG_COLS_MLP,
        )
        cfg = TrainConfig(
            lookback=lookback_window, horizon=1, batch_size=BATCH_SIZE,
            train_size=train_size, val_size=val_size,
            lr=LEARNING_RATE_MLP, dropout=DROPOUT_MLP,
            epochs=EPOCHS, patience=PATIENCE,
            hidden_sizes=MLP_HIDDEN_SIZES, device=str(device),
        )
        t_train0 = time.time()
        mlp_model, _, train_losses, _, _ = train_mlp_forecaster(
            df=prod['df_p'], cfg=cfg, seed=BENCH_SEED, loss_type='mse',
            product_id=f"{product_id}_{store_id}",
            scaler=prod['scaler'], target_channel=0,
            target_col=TARGET_COL, exog_cols=EXOG_COLS_MLP,
            test_size=forecast_horizon, exog_scaler=mlp_exog_scaler,
            best_model_path=None, verbose=False,
        )
        train_s = time.time() - t_train0
        epochs_trained = len(train_losses)

        hist_target = np.concatenate([prod['train'], prod['val']]).astype(np.float32)
        hist_exog   = prod['df_p'][EXOG_COLS_MLP].iloc[
            prod['val_start_idx']:prod['test_start_idx']
        ].iloc[-lookback_window:].reset_index(drop=True)
        future_exog = prod['df_p'][EXOG_COLS_MLP].iloc[
            prod['test_slice']
        ].reset_index(drop=True)

        t_infer0 = time.time()
        recursive_inference_dynamic_exog(
            model=mlp_model,
            target_scaler=prod['scaler'],
            exog_scaler=mlp_exog_scaler,
            exog_cols=EXOG_COLS_MLP,
            history_target_unscaled=hist_target,
            history_exog_unscaled=hist_exog,
            future_exog_unscaled=future_exog,
            target_channel=0,
            device=str(device),
        )
        infer_s = time.time() - t_infer0
        peak_gpu_mb = _read_peak_gpu_mb()
        peak_ram_mb = max(0.0, _proc_rss_mb() - _ram_baseline_mb)
        s_per_epoch = train_s / epochs_trained if epochs_trained else 0.0
        return 0.0, train_s, infer_s, peak_gpu_mb, peak_ram_mb, epochs_trained, s_per_epoch

    # ── Graph variants (gcn / gat / graph2vec  ×  lstm / mlp) ────────────────
    head      = _variant_head(variant)
    exog_cols = EXOG_COLS_LSTM if head == 'lstm' else EXOG_COLS_MLP
    eb        = prod['exog_by_head'][head]

    t_build0 = time.time()
    (pyg_train, pyg_val, pyg_seed_graphs, pyg_future_graphs,
     in_channels, graph2vec_model, current_df_wide) = build_graphs_for_variant(
        variant, shared, prod, product_id,
    )
    build_s = time.time() - t_build0

    DatasetClass = GCN_LSTMTimeSeriesDataset if head == 'lstm' else GCNMLPTimeSeriesDataset
    collate_fn   = collate_pyg_ts_lstm       if head == 'lstm' else collate_pyg_ts_mlp
    train_fn     = train_gcn_lstm_model      if head == 'lstm' else train_gcn_mlpmodel

    pin  = torch.cuda.is_available()
    tr_ds = DatasetClass(
        prod['train_scaled'], eb['train'] if exog_cols else None,
        lookback_window, pyg_train, graph_window_size=WINDOW_SIZE,
    )
    va_ds = DatasetClass(
        prod['val_scaled'], eb['val'] if exog_cols else None,
        lookback_window, pyg_val, graph_window_size=WINDOW_SIZE,
    )
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=False,
                       pin_memory=pin, collate_fn=collate_fn)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       pin_memory=pin, collate_fn=collate_fn)

    ts_input_size = 1 + (len(exog_cols) if exog_cols else 0)
    _seed_everything(BENCH_SEED)
    model = build_forecaster(variant, in_channels, ts_input_size).to(device)
    lr  = LEARNING_RATE_LSTM if head == 'lstm' else LEARNING_RATE_MLP
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=PATIENCE // 3)

    t_train0 = time.time()
    model, train_losses, _, _, _ = train_fn(
        seed=BENCH_SEED, epochs=EPOCHS, model=model,
        train_loader=tr_ld, val_loader=va_ld,
        criterion=nn.MSELoss(), criterion2=nn.MSELoss(),
        optimizer=opt, device=device, best_model_path=None,
        scheduler=sch, patience=PATIENCE,
        diag_csv_path=None, diag_meta={},
    )
    train_s = time.time() - t_train0
    epochs_trained = len(train_losses)

    # ── Build ts_seed for inference ───────────────────────────────────────────
    if exog_cols:
        exog_seed_rows = np.vstack([
            eb['val'][-(lookback_window - 1):],
            eb['test'][0:1],
        ])
        ts_seed = np.column_stack([
            prod['val_scaled'][-lookback_window:].reshape(-1, 1),
            exog_seed_rows,
        ]).astype(np.float32)
    else:
        ts_seed = prod['val_scaled'][-lookback_window:].reshape(-1, 1).astype(np.float32)

    lag_col_indices = (
        {exog_cols.index(n): k for n, k in LAG_K_BY_NAME.items() if n in exog_cols}
        if exog_cols else {}
    )
    rolling_mean_excl_col_indices = {}
    if exog_cols:
        for i, name in enumerate(exog_cols):
            if name.startswith(ROLLING_MEAN_EXCL_PREFIX):
                try:
                    rolling_mean_excl_col_indices[i] = int(
                        name[len(ROLLING_MEAN_EXCL_PREFIX):]
                    )
                except ValueError:
                    pass

    target_history_unscaled = np.concatenate(
        [prod['train'], prod['val']]
    ).astype(np.float32)

    metric_type    = infer_metric_type(METRIC)
    T_global       = current_df_wide.shape[1]
    product_offset = T_global - len(prod['df_p'])

    initial_target_window = None
    target_z_scaler       = None
    if product_id in current_df_wide.index:
        tw_start = max(0, product_offset + prod['test_start_idx'] - WINDOW_SIZE)
        initial_target_window = current_df_wide.loc[product_id].values[
            tw_start: product_offset + prod['test_start_idx']
        ].astype(np.float32)
        if metric_type == 'distance' and product_id in shared['product_scalers']:
            target_z_scaler = shared['product_scalers'][product_id]

    t_infer0 = time.time()
    if variant.startswith('graph2vec'):
        recursive_forecast_graph2vec_perstep(
            model=model,
            ts_seed=ts_seed,
            initial_graphs=pyg_seed_graphs,
            exog_test_scaled=eb['test'] if exog_cols else None,
            scaler=prod['scaler'],
            horizon=forecast_horizon,
            device=device,
            graph2vec_model=graph2vec_model,
            df_wide=current_df_wide,
            cat_labels=shared['cat_labels_dict'],
            target_id=product_id,
            metric=METRIC,
            fixed_threshold=PARAM_VAL,
            graph_window_size=WINDOW_SIZE,
            first_forecast_col=product_offset + prod['test_start_idx'],
            target_seed_window_dfscale=initial_target_window,
            enable_edges_within_star=ENABLE_EDGES,
            enable_second_degree=ENABLE_2ND,
            target_df_scaler=target_z_scaler,
            target_history_unscaled=target_history_unscaled if exog_cols else None,
            lag_col_indices=lag_col_indices if exog_cols else None,
            rolling_mean_excl_col_indices=rolling_mean_excl_col_indices if exog_cols else None,
            exog_scaler=eb['scaler'] if exog_cols else None,
        )
    elif head == 'lstm':
        _recursive_forecast_gcn_lstm_perstep(
            model=model,
            ts_seed=ts_seed,
            initial_graphs=pyg_seed_graphs,
            future_graphs=pyg_future_graphs,
            exog_test_scaled=eb['test'] if exog_cols else None,
            scaler=prod['scaler'],
            horizon=forecast_horizon,
            device=device,
            target_history_unscaled=target_history_unscaled if exog_cols else None,
            lag_col_indices=lag_col_indices if exog_cols else None,
            rolling_mean_excl_col_indices=rolling_mean_excl_col_indices if exog_cols else None,
            exog_scaler=eb['scaler'] if exog_cols else None,
            initial_target_window=initial_target_window,
            target_z_scaler=target_z_scaler,
            node_feature_mode=NODE_FEAT_MODE,
        )
    else:
        _recursive_forecast_gcn_mlp_perstep(
            model=model,
            ts_seed=ts_seed,
            initial_graphs=pyg_seed_graphs,
            future_graphs=pyg_future_graphs,
            exog_test_scaled=eb['test'] if exog_cols else None,
            scaler=prod['scaler'],
            horizon=forecast_horizon,
            device=device,
            target_history_unscaled=target_history_unscaled if exog_cols else None,
            lag_col_indices=lag_col_indices if exog_cols else None,
            rolling_mean_excl_col_indices=rolling_mean_excl_col_indices if exog_cols else None,
            exog_scaler=eb['scaler'] if exog_cols else None,
            exog_cols=exog_cols if exog_cols else None,
            node_feature_mode=NODE_FEAT_MODE,
        )
    infer_s = time.time() - t_infer0
    peak_gpu_mb = _read_peak_gpu_mb()
    peak_ram_mb = max(0.0, _proc_rss_mb() - _ram_baseline_mb)
    s_per_epoch = train_s / epochs_trained if epochs_trained else 0.0
    return build_s, train_s, infer_s, peak_gpu_mb, peak_ram_mb, epochs_trained, s_per_epoch


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _load_checkpoint(csv_path):
    """Read an existing results CSV and return per-variant resume state.

    Returns
    -------
    completed : dict[str, set[tuple]]
        variant -> {(product_id, store_id), ...} already written to CSV.
    cum_state : dict[str, dict]
        variant -> last cumulative values seen in the CSV so that the new
        run can continue accumulating without double-counting.
    """
    completed = {}
    cum_state = {}
    if not os.path.exists(csv_path):
        return completed, cum_state
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get('variant', '')
                if not v:
                    continue
                try:
                    pid = int(row['product_id'])
                    sid = int(row['store_id'])
                except (KeyError, ValueError):
                    continue
                completed.setdefault(v, set()).add((pid, sid))
                cum_state[v] = {
                    'cum_build':  float(row.get('cum_build_s',  0)),
                    'cum_train':  float(row.get('cum_train_s',  0)),
                    'cum_infer':  float(row.get('cum_infer_s',  0)),
                    'max_gpu_mb': float(row.get('cum_peak_gpu_mb', 0)),
                    'max_ram_mb': float(row.get('cum_peak_ram_mb', 0)),
                    'n_done':     int(row.get('n', 0)),
                }
    except Exception as exc:
        print(f"[checkpoint] Could not read {csv_path}: {exc}. Starting fresh.")
        return {}, {}
    return completed, cum_state


# ── Benchmark driver ──────────────────────────────────────────────────────────
def main():
    max_n  = max(PRODUCT_COUNTS)
    shared = prepare_shared()

    eligible = []
    for pid, sid in shared['products']:
        if len(eligible) >= max_n:
            break
        if prepare_product(shared, pid, sid) is not None:
            eligible.append((pid, sid))
    print(f"\nBenchmarking on {len(eligible)} eligible products (max requested: {max_n}).")

    cutoffs = sorted(n for n in PRODUCT_COUNTS if n <= len(eligible))
    summary_rows = []  # (variant, n, cum_build, cum_train, cum_infer, total, mean_pp, max_gpu, max_ram) — at cutoffs only, for plots

    completed, cum_state = _load_checkpoint(RESULTS_CSV)
    resume = bool(completed)
    if resume:
        total_done = sum(len(v) for v in completed.values())
        print(f"\n[resume] Found {RESULTS_CSV} with {total_done} completed (variant, product) rows.")
        print(f"[resume] Variants already started: {list(completed.keys())}")

    csv_mode = 'a' if resume else 'w'
    with open(RESULTS_CSV, csv_mode, newline='') as csv_f:
        writer = csv.writer(csv_f)
        if not resume:
            writer.writerow([
                "variant", "product_id", "store_id", "n",
                "build_s", "train_s", "infer_s",
                "cum_build_s", "cum_train_s", "cum_infer_s",
                "product_peak_gpu_mb", "product_peak_ram_mb",
                "cum_peak_gpu_mb", "cum_peak_ram_mb",
                "epochs_trained", "s_per_epoch",
            ])
            csv_f.flush()

        for variant in BENCH_VARIANTS:
            done_set = completed.get(variant, set())
            state    = cum_state.get(variant, {})
            cum_build  = state.get('cum_build',  0.0)
            cum_train  = state.get('cum_train',  0.0)
            cum_infer  = state.get('cum_infer',  0.0)
            max_gpu_mb = state.get('max_gpu_mb', 0.0)
            max_ram_mb = state.get('max_ram_mb', 0.0)
            n_done     = state.get('n_done',     0)

            print(f"\n{'#'*70}\n# VARIANT: {variant}\n{'#'*70}")
            if done_set:
                print(f"  [resume] skipping {len(done_set)} already-done products (resuming from n={n_done})")

            for pid, sid in eligible:
                if (pid, sid) in done_set:
                    continue

                try:
                    res = run_one(variant, shared, pid, sid)
                except Exception as exc:
                    print(f"  [{variant}] product {pid}/{sid} FAILED: {exc}")
                    traceback.print_exc()
                    res = None

                if res is not None:
                    b, t, inf, gpu_mb, ram_mb, epochs, spe = res
                    cum_build  += b
                    cum_train  += t
                    cum_infer  += inf
                    max_gpu_mb  = max(max_gpu_mb, gpu_mb)
                    max_ram_mb  = max(max_ram_mb, ram_mb)
                    n_done     += 1
                    print(
                        f"  [{variant}] {n_done}/{len(eligible)}"
                        f"  build={b:6.2f}s  train={t:7.2f}s  infer={inf:6.2f}s"
                        f"  gpu={gpu_mb:7.1f}MB  ram={ram_mb:7.1f}MB"
                        f"  epochs={epochs:4d}  s/ep={spe:.4f}"
                        f"  (cum_train={cum_train:8.2f}s  cum_infer={cum_infer:7.2f}s)"
                    )
                    writer.writerow([
                        variant, pid, sid, n_done,
                        round(b, 3), round(t, 3), round(inf, 3),
                        round(cum_build, 3), round(cum_train, 3), round(cum_infer, 3),
                        round(gpu_mb, 2), round(ram_mb, 2),
                        round(max_gpu_mb, 2), round(max_ram_mb, 2),
                        epochs, round(spe, 6),
                    ])
                    csv_f.flush()

                if n_done in cutoffs:
                    cutoffs_seen = [c for c in cutoffs if c <= n_done]
                    if cutoffs_seen and cutoffs_seen[-1] == n_done:
                        mean_pp = (cum_train + cum_infer) / n_done if n_done else float('nan')
                        total   = cum_build + cum_train + cum_infer
                        summary_rows.append((
                            variant, n_done,
                            round(cum_build, 3), round(cum_train, 3), round(cum_infer, 3),
                            round(total, 3), round(mean_pp, 4),
                            round(max_gpu_mb, 2), round(max_ram_mb, 2),
                        ))
                        print(
                            f"  >>> CUTOFF N={n_done}: "
                            f"build={cum_build:.2f}s  train={cum_train:.2f}s  "
                            f"infer={cum_infer:.2f}s  total={total:.2f}s  "
                            f"peak_gpu={max_gpu_mb:.1f}MB  peak_ram={max_ram_mb:.1f}MB"
                        )

    print(f"\nSaved scalability results -> {RESULTS_CSV}")

    # ── Print summary table ───────────────────────────────────────────────────
    hdr = (f"{'variant':<18}{'N':>6}{'build_s':>10}"
           f"{'train_s':>10}{'infer_s':>10}{'total_s':>10}{'mean/prod':>11}"
           f"{'peak_gpu_MB':>13}{'peak_ram_MB':>13}")
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for variant, n, b, t, inf, tot, mean, gpu_mb, ram_mb in summary_rows:
        print(
            f"{variant:<18}{n:>6}{b:>10.2f}{t:>10.2f}{inf:>10.2f}"
            f"{tot:>10.2f}{mean:>11.3f}{gpu_mb:>13.1f}{ram_mb:>13.1f}"
        )

    plot_scalability(summary_rows)
    plot_memory(summary_rows)


# ── Scalability plot ──────────────────────────────────────────────────────────
def plot_scalability(rows):
    """
    Three-panel figure:
      left   – cumulative total time (build + train + infer) vs n_products
      centre – cumulative train time vs n_products
      right  – cumulative inference time vs n_products

    Each line is one model variant; markers are drawn only at the exact
    PRODUCT_COUNTS cutoff points and annotated with the product count.
    """
    from collections import defaultdict

    # ── Collect per-variant series ────────────────────────────────────────────
    series = defaultdict(lambda: dict(n=[], total=[], train=[], infer=[]))
    for variant, n, build, train, infer, total, _mean, _gpu_mb, _ram_mb in rows:
        series[variant]['n'].append(n)
        series[variant]['total'].append(total)
        series[variant]['train'].append(train)
        series[variant]['infer'].append(infer)

    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    ]
    markers  = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    variants = [v for v in BENCH_VARIANTS if v in series]

    panels = [
        ('total', 'Cumulative total time (s)',      'Total  (build + train + infer)'),
        ('train', 'Cumulative train time (s)',       'Train'),
        ('infer', 'Cumulative inference time (s)',   'Inference'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    fig.suptitle('Scalability analysis — time vs number of products', fontsize=14, y=1.01)

    for ax, (key, ylabel, title) in zip(axes, panels):
        for idx, variant in enumerate(variants):
            xs = series[variant]['n']
            ys = series[variant][key]
            color  = palette[idx % len(palette)]
            marker = markers[idx % len(markers)]

            ax.plot(xs, ys,
                    color=color, marker=marker,
                    linewidth=2, markersize=9,
                    markerfacecolor=color, markeredgecolor='white',
                    markeredgewidth=0.8,
                    label=variant, zorder=3)

            # Annotate each cutoff with its n value
            for x, y in zip(xs, ys):
                ax.annotate(
                    str(x),
                    xy=(x, y),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=7, color=color,
                    fontweight='bold',
                )

        # Vertical guide-lines at each product-count cutoff
        for n in PRODUCT_COUNTS:
            ax.axvline(n, color='grey', linewidth=0.5, linestyle=':', alpha=0.6, zorder=1)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Number of products', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(PRODUCT_COUNTS)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)

    # Single shared legend below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=len(variants),
               fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, "scalability_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved scalability plot   -> {plot_path}")


# ── Memory plot ───────────────────────────────────────────────────────────────
def plot_memory(rows):
    """
    Two-panel figure:
      left  – peak GPU memory (MB) vs n_products
      right – peak RAM increase (MB) vs n_products

    Each line is one model variant; markers drawn at PRODUCT_COUNTS cutoffs.
    """
    from collections import defaultdict

    series = defaultdict(lambda: dict(n=[], gpu=[], ram=[]))
    for row in rows:
        variant, n = row[0], row[1]
        gpu_mb, ram_mb = row[7], row[8]
        series[variant]['n'].append(n)
        series[variant]['gpu'].append(gpu_mb)
        series[variant]['ram'].append(ram_mb)

    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    ]
    markers  = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    variants = [v for v in BENCH_VARIANTS if v in series]

    panels = [
        ('gpu', 'Peak GPU memory (MB)',    'Peak GPU Memory'),
        ('ram', 'Peak RAM increase (MB)',  'Peak RAM Increase'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle('Scalability analysis — peak memory vs number of products', fontsize=14, y=1.01)

    for ax, (key, ylabel, title) in zip(axes, panels):
        for idx, variant in enumerate(variants):
            xs = series[variant]['n']
            ys = series[variant][key]
            color  = palette[idx % len(palette)]
            marker = markers[idx % len(markers)]

            ax.plot(xs, ys,
                    color=color, marker=marker,
                    linewidth=2, markersize=9,
                    markerfacecolor=color, markeredgecolor='white',
                    markeredgewidth=0.8,
                    label=variant, zorder=3)

            for x, y in zip(xs, ys):
                ax.annotate(
                    str(x),
                    xy=(x, y),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=7, color=color,
                    fontweight='bold',
                )

        for n in PRODUCT_COUNTS:
            ax.axvline(n, color='grey', linewidth=0.5, linestyle=':', alpha=0.6, zorder=1)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Number of products', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(PRODUCT_COUNTS)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=len(variants),
               fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, "scalability_memory.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved memory plot        -> {plot_path}")


if __name__ == '__main__':
    main()
