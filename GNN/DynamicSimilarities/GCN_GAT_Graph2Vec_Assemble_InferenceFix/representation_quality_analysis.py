"""
Representation-quality analysis for the GCN / GAT / Graph2Vec assemble,
FOCUSED on a single arbitrarily-chosen (product, seed).

Answers "did the graph embeddings capture meaningful temporal/structural market
dynamics?" via three analyses on one selected product (set SELECTED_PRODUCT /
SELECTED_SEED below):

  1. Topological ablation (ablate_z) — full model (graph embedding z concatenated)
     vs the matched graph-free model (z removed) for each graph variant, on the
     selected seed.  A large RMSE gap means z carried exploitable information the
     temporal head alone could not.  GCN/GAT/Graph2Vec collapse to one graph-free
     model per head when z is removed, so the ablated reference is trained once
     per head (via gcn_lstm / gcn_mlp) and reused.  Single-seed DELTA (descriptive,
     no p-value).

  2. Latent-space clustering (t-SNE / UMAP) — reduce the per-window Graph2Vec
     embeddings to 2D, colour by Season and Macro-Event.  Tight event clusters
     mean the embedding captured seasonal/structural demand shifts.

  3. Temporal stability (cosine similarity) — cos(z_t, z_{t+1}) over adjacent
     sliding windows, plotted over time with macro-events / promotions marked.

This module re-imports the SAME underlying routines the runner uses (datasets,
models, training functions, the Graph2Vec embedding pipeline and the recursive
inference routines) directly from their source modules, and mirrors the runner's
current hyper-parameters (resolve_learning_rate, per-head scheduler toggles) so
the ablation numbers track the real experiments.

Output (under representation_quality/product_<id>_<store>/seed_<seed>/)
    ablation_summary.csv               : full vs ablated RMSE/MAE/POCID per variant.
    latent_space_{tsne,umap}.pdf       : 2D embedding scatter (by season / event).
    latent_space_{tsne,umap}_coords.csv: the 2D coordinates + labels.
    temporal_stability_cosine.pdf/.csv : cos(z_t, z_{t+1}) over time.
"""

from __future__ import annotations

import os
import sys
import csv
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ── Paths & sys.path setup (so the sibling/parent modules import cleanly) ────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)                                          # local modules
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))     # DynamicSimilarities/

# ── The SAME routines the runner uses — imported directly from their source ──
from utils import (
    generate_exogenous_features, compute_metrics, neighbourhood_graph,
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
    _recursive_forecast_gcn_mlp_perstep, _recursive_forecast_gcn_lstm_perstep,
)
from generate_graph2vecwithadaptativethreshold import load_or_generate_embeddings
from graph2vec_assemble import (
    recursive_forecast_graph2vec_perstep, embedding_to_single_node_data,
)
from ablationlstm import AblationLSTMForecaster
from ablationmlp import AblationMLPForecaster
from train_methods import train_gcn_lstm_model, train_gcn_mlpmodel

from LSTMBaseline.dataset import TimeSeriesDataset as LSTMBaselineDataset
from LSTMBaseline.lstm import LSTM as LSTMBaseline
from LSTMBaseline.lstm_train import train_model as train_lstm_baseline
from LSTMBaseline.lstm_inference import recursive_inference as recursive_forecast_lstm_baseline

from MLP_Baseline.train import train_mlp_forecaster, TrainConfig
from MLP_Baseline.inference import recursive_inference_dynamic_exog
from MLP_Baseline.utils import ExogenousScaler


# ── Metric typing (copied from the runner) ───────────────────────────────────
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
    raise ValueError(f"Metric {metric} not supported.")


# ── Constants (same defaults as the runner) ──────────────────────────────────
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/data_andre_classified.feather'))
DATE_COL = 'date'
TARGET_COL = 'value'

val_size = 61
forecast_horizon = 153
train_size = 761 - val_size - forecast_horizon
lookback_window = 30
BATCH_SIZE = 32

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
    "dom_sin", "dom_cos", "wom_sin", "wom_cos", "month_sin", "month_cos",
    "quarter_sin", "quarter_cos", "woy_sin", "woy_cos",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "rolling_mean_excl_7",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_bridge_day",
]
# Union — used by generate_exogenous_features so every feature is computed once.
EXOG_COLS = list(dict.fromkeys(EXOG_COLS_LSTM + EXOG_COLS_MLP))

# Similarity/graph grid, kept list-shaped like the runner; the analysis uses the
# first entry of each (single config).
grid_configs = [
    #{'metric': 'spearman', 'thresholds': [0.634]},
    {'metric': 'cid', 'thresholds': [4.17]},
]
window_sizes              = [30]
step_sizes                = [1]
enable_edges_opts         = [True]
enable_second_degree_opts = [False]
node_feature_modes        = ['catch24_minmaxlast']

USE_RESIDUALS  = False
MODEL_TYPE     = 'ridge'
EPOCHS         = 1000
PATIENCE       = 100
LEARNING_RATE_LSTM     = 0.001
LEARNING_RATE_MLP      = 0.0001
LEARNING_RATE_GAT_LSTM = 0.0005      # gat_lstm only (mirrors the runner)
HIDDEN_SIZE    = 32
NUM_LAYERS     = 1
DROPOUT_LSTM   = 0.0
DROPOUT_MLP    = 0.2
D_G            = 16

# Per-head ReduceLROnPlateau toggle (mirrors the runner: MLP head fixed-LR).
USE_LR_SCHEDULER_LSTM = True
USE_LR_SCHEDULER_MLP  = False

# Model-architecture grid bits used by build_forecaster.
ATTENTION_HEADS  = 4            # GAT attention heads (encoder layer 1)
MLP_HIDDEN_SIZES = (128, 64)    # TimeDistributed-MLP head widths

# Exog columns whose value at day t depends on the target's own history — must
# be recomputed from the rolling forecast during recursive inference.
LAG_K_BY_NAME = {"lag_1": 1, "lag_7": 7, "lag_30": 30}
ROLLING_MEAN_EXCL_PREFIX = "rolling_mean_excl_"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Single similarity/threshold config + grid point taken from the grid above ─
_CFG          = grid_configs[0]
METRIC        = _CFG['metric']
_THRESHOLDS   = _CFG.get('thresholds', [None])
IS_THRESHOLD  = _THRESHOLDS not in (None, [None])
PARAM_VAL     = (_THRESHOLDS[0] if IS_THRESHOLD else _CFG.get('percentiles', [None])[0])
WINDOW_SIZE   = window_sizes[0]
STEP_SIZE     = step_sizes[0]
ENABLE_EDGES  = enable_edges_opts[0]
ENABLE_2ND    = enable_second_degree_opts[0]
NODE_FEAT_MODE = node_feature_modes[0]

# Metrics that operate in z-score space (target wide matrix is StandardScaler'd).
_DISTANCE_METRICS = ['euclidean', 'manhattan', 'hamming', 'amplitude_offset',
                     'slope_consistency', 'phase_invariance', 'dtw', 'cid',
                     'lorentzian', 'sbd', 'msm', 'edr', 'lcss']


# ════════════════════════════════════════════════════════════════════════════
#  ANALYSIS CONFIGURATION — set the product/seed and which analyses to run
# ════════════════════════════════════════════════════════════════════════════
# Two ways to choose what to analyse:
#   1. SELECTED_PRODUCT_SEED_PAIRS (preferred) — an explicit list of
#      ``(item_id, seed)`` pairs.  Each item_id is paired with EXACTLY its own
#      seed (not the full product × seed cross-product); the store is constant
#      ``STORE_ID``.  Set to None/[] to fall back to the single-pair mode below.
#   2. SELECTED_PRODUCT / SELECTED_SEED — a single (item_id, store_id) + seed,
#      used only when SELECTED_PRODUCT_SEED_PAIRS is empty.
STORE_ID = 6269                      # store is constant across all items here
SELECTED_PRODUCT_SEED_PAIRS = [
    # --- LSTM Improvements ---
    (907969, 5219788),    # Best lstm_baseline
    (907969, 618626816),  # Best gcn_lstm
    (915640, 907969),     # Best mlp_baseline
    (915640, 26008),      # Best gcn_lstm
    (213636, 13451285),   # lstm_baseline & gcn_lstm
    (921558, 213626),     # lstm_baseline
    (921558, 907969),     # graph2vec_lstm
    # --- MLP Improvements ---
    (915640, 1000),       # Best mlp_baseline
    (921558, 1000),       # mlp_baseline & gat_mlp
    (26924, 1000),        # mlp_baseline
    (26924, 907969),      # gat_mlp
    (13544, 13451285),    # mlp_baseline
    (13544, 23616558),    # gat_mlp
    (26076, 1000),        # mlp_baseline
    (26076, 618626816),   # gcn_mlp
]

# SELECTED_PRODUCT : (item_id, store_id) tuple, or None for the first eligible.
# SELECTED_SEED    : single seed for embedding generation + training.
# These are MUTATED per pair in main() so the seed-dependent helpers
# (build_graphs_for_variant, run_variant, extract_graph2vec_embeddings, ...) that
# read the SELECTED_SEED / SELECTED_PRODUCT globals pick up the current pair.
SELECTED_PRODUCT = None
SELECTED_SEED    = 42

RUN_ABLATION = True     # Topological ablation: full (z) vs ablated (z=0)
RUN_TSNE     = True     # Latent-space clustering of Graph2Vec embeddings
RUN_COSINE   = True     # Temporal stability: cos(z_t, z_{t+1}) over time

# Graph variants compared against their head's z-ablated counterpart.
ABLATION_VARIANTS = [
    'gcn_lstm', 'gat_lstm', 'graph2vec_lstm',
    'gcn_mlp',  'gat_mlp',  'graph2vec_mlp',
]
ABLATION_INCLUDE_BASELINES = True       # add the pure baselines as context rows

# Dimensionality-reduction methods.  'tsne' always runs (sklearn); 'umap' is
# attempted only if the package is importable.
DR_METHODS = ['tsne', 'umap']

OUT_BASE = os.path.join(SCRIPT_DIR, "representation_quality")


def _variant_head(variant):
    """'lstm' or 'mlp' — the temporal head implied by a variant name."""
    return 'mlp' if variant.endswith('mlp') else 'lstm'


def resolve_learning_rate(variant):
    """Per-variant LR (mirrors the runner): gat_lstm gets the gentler GAT rate."""
    head = _variant_head(variant)
    if variant.startswith('gat') and head == 'lstm':
        return LEARNING_RATE_GAT_LSTM
    return LEARNING_RATE_LSTM if head == 'lstm' else LEARNING_RATE_MLP


def build_forecaster(variant, ablate_z, in_channels, ts_input_size):
    """Construct the forecaster for ``variant`` (copied from the runner)."""
    head = _variant_head(variant)
    dropout = DROPOUT_LSTM if head == 'lstm' else DROPOUT_MLP
    if ablate_z:
        if head == 'lstm':
            return AblationLSTMForecaster(
                lstm_input_size=ts_input_size, lstm_hidden=HIDDEN_SIZE,
                lstm_layers=NUM_LAYERS, horizon=1, dropout=dropout,
            )
        return AblationMLPForecaster(
            ts_input_size=ts_input_size, hidden_sizes=MLP_HIDDEN_SIZES,
            dropout=dropout,
        )
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
    raise ValueError(f"Unknown model variant: {variant!r}")


# ── Per-day alignment helpers (copied from the runner) ───────────────────────
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
    """Per-window PyG list -> per-day list of length T; aligned[t] = graph built
    strictly before day t (days t-W .. t-1).  Pads by W to guarantee no leakage."""
    if step_size != 1:
        raise NotImplementedError("Per-day alignment helper currently assumes step_size=1")
    pad = _make_pad_graph(pyg_windows[0])
    aligned = [pad] * window_size + list(pyg_windows)
    if len(aligned) < T:
        aligned += [aligned[-1]] * (T - len(aligned))
    else:
        aligned = aligned[:T]
    return aligned


def _seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(test, forecast):
    """Full metric dict over the valid (non-NaN) forecast steps."""
    arr = np.asarray(forecast, dtype=float)
    valid = ~np.isnan(arr)
    if not valid.any():
        return dict(rmse=float('nan'), mae=float('nan'), bias=float('nan'),
                    score=float('nan'), pocid=float('nan'), forecast=arr)
    rmse, mae, bias, score, pocid = compute_metrics(test[valid], arr[valid])
    return dict(rmse=float(rmse), mae=float(mae), bias=float(bias),
                score=float(score), pocid=float(pocid), forecast=arr)


# ── Shared, product-independent setup (mirrors the runner's main()) ──────────
def prepare_shared():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_feather(DATA_PATH)
    if DATE_COL in df.index.names:
        df = df.reset_index(drop=True) if DATE_COL in df.columns else df.reset_index()
    if df.index.name == DATE_COL:
        df = df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, 'item_id', 'store_id']).reset_index(drop=True)
    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS)
    full_df = df.copy()

    products = (
        full_df[["item_id", "store_id"]]
        .drop_duplicates()
        .sort_values(["item_id", "store_id"])
        .apply(lambda r: (int(r["item_id"]), int(r["store_id"])), axis=1)
        .tolist()
    )
    pair_counts = full_df.groupby(["item_id", "store_id"]).size().to_dict()

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
    train_df_wide = df_wide_global.iloc[:, global_train_start_idx:global_val_start_idx]
    df_wide_scaled = df_wide_global.copy()
    for item_id_iter in df_wide_global.index:
        z = StandardScaler()
        z.fit(train_df_wide.loc[item_id_iter].values.reshape(-1, 1))
        product_scalers[item_id_iter] = z
        df_wide_scaled.loc[item_id_iter] = z.transform(
            df_wide_global.loc[item_id_iter].values.reshape(-1, 1)
        ).flatten()

    return dict(
        full_df=full_df, products=products, pair_counts=pair_counts,
        cat_labels_dict=cat_labels_dict, df_wide_global=df_wide_global,
        df_wide_scaled=df_wide_scaled, product_scalers=product_scalers,
        global_val_start_idx=global_val_start_idx,
    )


# ── Per-product data slice (mirrors the runner's per-product block) ──────────
def prepare_product(shared, product_id, store_id):
    full_df = shared['full_df']
    df_p = (
        full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)]
        .sort_values(DATE_COL).reset_index(drop=True)
    )
    required = forecast_horizon + val_size + train_size
    if len(df_p) < required:
        return None

    test_start_idx  = len(df_p) - forecast_horizon
    val_start_idx   = test_start_idx - val_size
    train_start_idx = val_start_idx - train_size
    train_slice = slice(train_start_idx, val_start_idx)
    val_slice   = slice(val_start_idx, test_start_idx)
    test_slice  = slice(test_start_idx, None)

    train = df_p[TARGET_COL][train_slice].values
    val   = df_p[TARGET_COL][val_slice].values
    test  = df_p[TARGET_COL][test_slice].values

    scaler = MinMaxScaler()
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
        df_p=df_p, train=train, val=val, test=test, scaler=scaler,
        train_scaled=train_scaled, val_scaled=val_scaled,
        train_start_idx=train_start_idx, val_start_idx=val_start_idx,
        test_start_idx=test_start_idx,
        train_slice=train_slice, val_slice=val_slice, test_slice=test_slice,
        exog_by_head=exog_by_head,
    )


# ── Build the per-day aligned graphs/embeddings for one graph variant ────────
def build_graphs_for_variant(variant, shared, prod, product_id):
    """Build everything a graph variant's train + inference needs."""
    metric_type = infer_metric_type(METRIC)
    current_df_wide = (shared['df_wide_scaled'] if METRIC in _DISTANCE_METRICS
                       else shared['df_wide_global'])
    T_global = current_df_wide.shape[1]
    df_p = prod['df_p']
    product_offset = T_global - len(df_p)
    graph2vec_model = None

    if variant.startswith('graph2vec'):
        (graph_embeddings, graph2vec_model, _csv, _bt, _et,
         fixed_threshold) = load_or_generate_embeddings(
            product_id=product_id, metric=METRIC, metric_type=metric_type,
            window_size=WINDOW_SIZE, step_size=STEP_SIZE,
            threshold=PARAM_VAL if IS_THRESHOLD else None,
            percentile=None if IS_THRESHOLD else PARAM_VAL,
            dimensions=D_G, enable_edges_within_star=ENABLE_EDGES,
            enable_second_degree=ENABLE_2ND, use_residuals=USE_RESIDUALS,
            model_type=MODEL_TYPE, seed=SELECTED_SEED, df=current_df_wide,
            cat_labels=shared['cat_labels_dict'],
            train_end_idx=shared['global_val_start_idx'], save_embeddings=False,
        )
        pyg_windows = [
            embedding_to_single_node_data(graph_embeddings[i], target_label=product_id)
            for i in range(len(graph_embeddings))
        ]
    else:  # gcn_* / gat_*
        compute_func = (compute_distances_1vsAll if metric_type == 'distance'
                        else compute_similarities_1vsAll)
        nx_graphs, fixed_threshold = neighbourhood_graph(
            product_id=product_id, df=current_df_wide, metric=METRIC,
            metric_type=metric_type, window_size=WINDOW_SIZE, compute_func=compute_func,
            threshold=PARAM_VAL if IS_THRESHOLD else None,
            percentile=None if IS_THRESHOLD else PARAM_VAL,
            step_size=STEP_SIZE, cat_labels=shared['cat_labels_dict'], plot_dir=None,
            residuals=USE_RESIDUALS, enable_edges_within_star=ENABLE_EDGES,
            enable_second_degree=ENABLE_2ND, train_end_idx=shared['global_val_start_idx'],
        )
        build_graphs_fn = (build_pyg_graphs_lstm if _variant_head(variant) == 'lstm'
                           else build_pyg_graphs_mlp)
        pyg_windows = build_graphs_fn(
            nx_graphs, current_df_wide, product_id,
            window_size=WINDOW_SIZE, step_size=STEP_SIZE, node_feature_mode=NODE_FEAT_MODE,
        )

    aligned = _align_pyg_windows_to_timeline(
        pyg_windows, window_size=WINDOW_SIZE, step_size=STEP_SIZE, T=T_global)
    tr_i, va_i, te_i = prod['train_start_idx'], prod['val_start_idx'], prod['test_start_idx']
    pyg_train = aligned[product_offset + tr_i: product_offset + va_i]
    pyg_val   = aligned[product_offset + va_i: product_offset + te_i]
    seed_start = product_offset + te_i - lookback_window
    pyg_seed_graphs = aligned[seed_start: product_offset + te_i]
    fut_start = product_offset + te_i
    pyg_future_graphs = aligned[fut_start: fut_start + forecast_horizon]

    return dict(
        pyg_train=pyg_train, pyg_val=pyg_val,
        pyg_seed_graphs=pyg_seed_graphs, pyg_future_graphs=pyg_future_graphs,
        in_channels=pyg_train[0].x.shape[1], fixed_threshold=fixed_threshold,
        current_df_wide=current_df_wide, product_offset=product_offset,
        metric_type=metric_type, graph2vec_model=graph2vec_model,
    )


# ── One full (variant, product): build + train + inference + metrics ─────────
def run_variant(variant, shared, prod, product_id, store_id, ablate_z=False):
    """Run the full pipeline for one variant on one product; return a metric dict.

    ``ablate_z=True`` swaps in the matched graph-free model (z removed); the graph
    branch's output is gone, so GCN/GAT/Graph2Vec collapse to the same model per
    head.  The graphs are still built/fed (the ablation model ignores them), so a
    gcn_* variant is the cheapest way to obtain the head's ablated reference.
    """
    # ── Baselines (graph-free; ablate_z not applicable) ──────────────────────
    if variant == 'lstm_baseline':
        cols = EXOG_COLS_LSTM
        eb = prod['exog_by_head']['lstm']
        _seed_everything(SELECTED_SEED)
        model = LSTMBaseline(
            input_size=1 + (len(cols) if cols else 0), hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, dropout=DROPOUT_LSTM,
        ).to(device)
        tr_ds = LSTMBaselineDataset(prod['train_scaled'], eb['train'] if cols else None, lookback_window)
        va_ds = LSTMBaselineDataset(prod['val_scaled'],   eb['val']   if cols else None, lookback_window)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=False)
        va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_LSTM)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=PATIENCE // 3)
        model, _, _, _, _ = train_lstm_baseline(
            epochs=EPOCHS, model=model, train_loader=tr_ld, val_loader=va_ld,
            exog_cols=cols, criterion=nn.MSELoss(), criterion2=nn.MSELoss(),
            optimizer=opt, device=device, best_model_path=None, scheduler=sch, patience=PATIENCE,
        )
        exog_test_raw = prod['df_p'][cols][prod['test_slice']].values if cols else None
        forecast, _ = recursive_forecast_lstm_baseline(
            model=model, test_start_idx=prod['test_start_idx'], seq_length=lookback_window,
            val_scaled=prod['val_scaled'],
            exog_val_scaled=eb['val'] if cols else None,
            exog_test_scaled=eb['test'] if cols else None,
            exog_test=exog_test_raw, scaler=prod['scaler'],
            exog_scaler=eb['scaler'] if cols else None, df_product=prod['df_p'], device=device,
            exog_cols=cols, forecast_window=forecast_horizon, strategy='best_val',
            item_id=product_id, store_id=store_id, loss_type='MSELoss', script_dir=SCRIPT_DIR,
        )
        return _metrics(prod['test'], forecast)

    if variant == 'mlp_baseline':
        cols = EXOG_COLS_MLP
        _seed_everything(SELECTED_SEED)
        mlp_exog_scaler = ExogenousScaler(continuous_strategy='minmax')
        mlp_exog_scaler.fit(prod['df_p'][cols].iloc[prod['train_slice']], cols)
        cfg = TrainConfig(
            lookback=lookback_window, horizon=1, batch_size=BATCH_SIZE,
            train_size=train_size, val_size=val_size, lr=LEARNING_RATE_MLP,
            dropout=DROPOUT_MLP, epochs=EPOCHS, patience=PATIENCE,
            hidden_sizes=MLP_HIDDEN_SIZES, device=str(device),
        )
        mlp_model, _, _, _, _ = train_mlp_forecaster(
            df=prod['df_p'], cfg=cfg, seed=SELECTED_SEED, loss_type='mse',
            product_id=f"{product_id}_{store_id}", scaler=prod['scaler'], target_channel=0,
            target_col=TARGET_COL, exog_cols=cols, test_size=forecast_horizon,
            exog_scaler=mlp_exog_scaler,
        )
        hist_target = np.concatenate([prod['train'], prod['val']]).astype(np.float32)
        hist_exog   = prod['df_p'][cols].iloc[prod['val_slice']].iloc[-lookback_window:].reset_index(drop=True)
        future_exog = prod['df_p'][cols].iloc[prod['test_slice']].reset_index(drop=True)
        forecast = recursive_inference_dynamic_exog(
            model=mlp_model, target_scaler=prod['scaler'], exog_scaler=mlp_exog_scaler,
            exog_cols=cols, history_target_unscaled=hist_target,
            history_exog_unscaled=hist_exog, future_exog_unscaled=future_exog,
            target_channel=0, device=str(device),
        )
        return _metrics(prod['test'], forecast)

    # ── Graph variants (gcn/gat/graph2vec × lstm/mlp) ────────────────────────
    head      = _variant_head(variant)
    exog_cols = EXOG_COLS_LSTM if head == 'lstm' else EXOG_COLS_MLP
    eb        = prod['exog_by_head'][head]
    is_g2v    = variant.startswith('graph2vec')

    gb = build_graphs_for_variant(variant, shared, prod, product_id)

    if head == 'lstm':
        DatasetClass, collate_fn, train_fn = (
            GCN_LSTMTimeSeriesDataset, collate_pyg_ts_lstm, train_gcn_lstm_model)
    else:
        DatasetClass, collate_fn, train_fn = (
            GCNMLPTimeSeriesDataset, collate_pyg_ts_mlp, train_gcn_mlpmodel)

    pin = torch.cuda.is_available()
    tr_ds = DatasetClass(prod['train_scaled'], eb['train'] if exog_cols else None,
                         lookback_window, gb['pyg_train'], graph_window_size=WINDOW_SIZE)
    va_ds = DatasetClass(prod['val_scaled'], eb['val'] if exog_cols else None,
                         lookback_window, gb['pyg_val'], graph_window_size=WINDOW_SIZE)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin, collate_fn=collate_fn)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin, collate_fn=collate_fn)

    ts_input_size = 1 + (len(exog_cols) if exog_cols else 0)
    _seed_everything(SELECTED_SEED)
    model = build_forecaster(variant, ablate_z, gb['in_channels'], ts_input_size).to(device)
    model.ablate_z = ablate_z
    lr  = resolve_learning_rate(variant)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    use_sched = USE_LR_SCHEDULER_LSTM if head == 'lstm' else USE_LR_SCHEDULER_MLP
    sch = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                      patience=PATIENCE // 3)
           if use_sched else None)
    model, _, _, _, _ = train_fn(
        seed=SELECTED_SEED, epochs=EPOCHS, model=model,
        train_loader=tr_ld, val_loader=va_ld,
        criterion=nn.MSELoss(), criterion2=nn.MSELoss(),
        optimizer=opt, device=device, best_model_path=None,
        scheduler=sch, patience=PATIENCE, diag_csv_path=None, diag_meta={},
    )

    # ── Recursive inference (leak-safe; mirrors the runner) ──────────────────
    val_scaled       = prod['val_scaled']
    exog_test_scaled = eb['test']
    if exog_cols:
        exog_seed_rows = np.vstack([eb['val'][-(lookback_window - 1):], exog_test_scaled[0:1]])
        ts_seed = np.column_stack([
            val_scaled[-lookback_window:].reshape(-1, 1), exog_seed_rows,
        ]).astype(np.float32)
    else:
        ts_seed = val_scaled[-lookback_window:].reshape(-1, 1).astype(np.float32)

    lag_col_indices = {
        exog_cols.index(name): k
        for name, k in LAG_K_BY_NAME.items() if name in exog_cols
    }
    rolling_mean_excl_col_indices = {}
    for i, name in enumerate(exog_cols):
        if name.startswith(ROLLING_MEAN_EXCL_PREFIX):
            try:
                rolling_mean_excl_col_indices[i] = int(name[len(ROLLING_MEAN_EXCL_PREFIX):])
            except ValueError:
                continue
    target_history_unscaled = np.concatenate([prod['train'], prod['val']]).astype(np.float32)

    current_df_wide = gb['current_df_wide']
    product_offset  = gb['product_offset']
    metric_type     = gb['metric_type']
    initial_target_window = None
    target_z_scaler = None
    if product_id in current_df_wide.index:
        _tw_start = max(0, product_offset + prod['test_start_idx'] - WINDOW_SIZE)
        initial_target_window = current_df_wide.loc[product_id].values[
            _tw_start: product_offset + prod['test_start_idx']
        ].astype(np.float32)
        if metric_type == 'distance' and product_id in shared['product_scalers']:
            target_z_scaler = shared['product_scalers'][product_id]

    if is_g2v:
        forecast = recursive_forecast_graph2vec_perstep(
            model=model, ts_seed=ts_seed, initial_graphs=gb['pyg_seed_graphs'],
            exog_test_scaled=exog_test_scaled if exog_cols else None,
            scaler=prod['scaler'], horizon=forecast_horizon, device=device,
            graph2vec_model=gb['graph2vec_model'], df_wide=current_df_wide,
            cat_labels=shared['cat_labels_dict'], target_id=product_id, metric=METRIC,
            fixed_threshold=gb['fixed_threshold'], graph_window_size=WINDOW_SIZE,
            first_forecast_col=product_offset + prod['test_start_idx'],
            target_seed_window_dfscale=initial_target_window,
            enable_edges_within_star=ENABLE_EDGES, enable_second_degree=ENABLE_2ND,
            target_df_scaler=target_z_scaler,
            target_history_unscaled=target_history_unscaled if exog_cols else None,
            lag_col_indices=lag_col_indices if exog_cols else None,
            rolling_mean_excl_col_indices=rolling_mean_excl_col_indices if exog_cols else None,
            exog_scaler=eb['scaler'] if exog_cols else None,
        )
    elif head == 'lstm':
        forecast = _recursive_forecast_gcn_lstm_perstep(
            model=model, ts_seed=ts_seed, initial_graphs=gb['pyg_seed_graphs'],
            future_graphs=gb['pyg_future_graphs'],
            exog_test_scaled=exog_test_scaled if exog_cols else None,
            scaler=prod['scaler'], horizon=forecast_horizon, device=device,
            target_history_unscaled=target_history_unscaled if exog_cols else None,
            lag_col_indices=lag_col_indices if exog_cols else None,
            rolling_mean_excl_col_indices=rolling_mean_excl_col_indices if exog_cols else None,
            exog_scaler=eb['scaler'] if exog_cols else None,
            initial_target_window=initial_target_window, target_z_scaler=target_z_scaler,
            node_feature_mode=NODE_FEAT_MODE,
        )
    else:
        forecast = _recursive_forecast_gcn_mlp_perstep(
            model=model, ts_seed=ts_seed, initial_graphs=gb['pyg_seed_graphs'],
            future_graphs=gb['pyg_future_graphs'],
            exog_test_scaled=exog_test_scaled if exog_cols else None,
            scaler=prod['scaler'], horizon=forecast_horizon, device=device,
            target_history_unscaled=target_history_unscaled if exog_cols else None,
            lag_col_indices=lag_col_indices if exog_cols else None,
            rolling_mean_excl_col_indices=rolling_mean_excl_col_indices if exog_cols else None,
            exog_scaler=eb['scaler'] if exog_cols else None,
            exog_cols=exog_cols if exog_cols else None, node_feature_mode=NODE_FEAT_MODE,
        )

    return _metrics(prod['test'], forecast)


# ── Product selection ────────────────────────────────────────────────────────
def _resolve_selected_product(shared):
    """The (item_id, store_id) to analyse: SELECTED_PRODUCT if set + eligible,
    else the first product with enough history."""
    required = forecast_horizon + val_size + train_size
    eligible = [p for p in shared['products']
                if shared['pair_counts'].get(p, 0) >= required]
    if not eligible:
        raise RuntimeError("No product has enough history for the analysis.")
    if SELECTED_PRODUCT is not None:
        sel = (int(SELECTED_PRODUCT[0]), int(SELECTED_PRODUCT[1]))
        if shared['pair_counts'].get(sel, 0) < required:
            raise RuntimeError(f"SELECTED_PRODUCT {sel} missing or has insufficient history.")
        return sel
    print(f"SELECTED_PRODUCT is None -> using first eligible product {eligible[0]}.")
    return eligible[0]


# ── Analysis 1: Topological ablation (single-seed delta) ─────────────────────
def run_ablation_analysis(shared, prod, product_id, store_id, out_dir):
    print(f"\n{'='*72}\n# ANALYSIS 1 — Topological ablation (ablate_z), single-seed delta\n{'='*72}")

    rep_variant = {'lstm': 'gcn_lstm', 'mlp': 'gcn_mlp'}
    ablated_by_head = {}
    for head in sorted({_variant_head(v) for v in ABLATION_VARIANTS}):
        print(f"\n[ablated reference] head={head} (via {rep_variant[head]}, z=0)")
        ablated_by_head[head] = run_variant(rep_variant[head], shared, prod,
                                            product_id, store_id, ablate_z=True)
        print(f"  -> ablated RMSE={ablated_by_head[head]['rmse']:.4f}")

    rows = []
    for variant in ABLATION_VARIANTS:
        head = _variant_head(variant)
        print(f"\n[full] {variant} (z concatenated)")
        full = run_variant(variant, shared, prod, product_id, store_id, ablate_z=False)
        abl  = ablated_by_head[head]
        d_rmse = abl['rmse'] - full['rmse']                 # >0 => graph helps
        pct = 100.0 * d_rmse / abl['rmse'] if abl['rmse'] else float('nan')
        rows.append(dict(
            variant=variant, head=head,
            full_rmse=full['rmse'], ablated_rmse=abl['rmse'],
            delta_rmse=d_rmse, pct_rmse_improvement=pct,
            full_mae=full['mae'], ablated_mae=abl['mae'],
            full_pocid=full['pocid'], ablated_pocid=abl['pocid'],
        ))
        print(f"  full RMSE={full['rmse']:.4f}  ablated RMSE={abl['rmse']:.4f}  "
              f"delta={d_rmse:+.4f} ({pct:+.1f}%)")

    if ABLATION_INCLUDE_BASELINES:
        for bl in ('lstm_baseline', 'mlp_baseline'):
            print(f"\n[baseline] {bl}")
            m = run_variant(bl, shared, prod, product_id, store_id)
            rows.append(dict(
                variant=bl, head=_variant_head(bl),
                full_rmse=m['rmse'], ablated_rmse=float('nan'),
                delta_rmse=float('nan'), pct_rmse_improvement=float('nan'),
                full_mae=m['mae'], ablated_mae=float('nan'),
                full_pocid=m['pocid'], ablated_pocid=float('nan'),
            ))
            print(f"  RMSE={m['rmse']:.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "ablation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved ablation summary -> {csv_path}")
    print("\n--- Ablation (z) | lower RMSE better; delta>0 means the graph helps ---")
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(df.to_string(index=False))
    return df


# ── Graph2Vec embedding extraction (shared by analyses 2 & 3) ────────────────
def extract_graph2vec_embeddings(shared, product_id):
    """Per-window Graph2Vec embeddings for the selected product + their window
    end-dates.  Window i covers days [i, i+W-1]; attributed to its end day."""
    metric_type = infer_metric_type(METRIC)
    current_df_wide = (shared['df_wide_scaled'] if METRIC in _DISTANCE_METRICS
                       else shared['df_wide_global'])
    (graph_embeddings, _g2v, _csv, _bt, _et,
     _thr) = load_or_generate_embeddings(
        product_id=product_id, metric=METRIC, metric_type=metric_type,
        window_size=WINDOW_SIZE, step_size=STEP_SIZE,
        threshold=PARAM_VAL if IS_THRESHOLD else None,
        percentile=None if IS_THRESHOLD else PARAM_VAL,
        dimensions=D_G, enable_edges_within_star=ENABLE_EDGES,
        enable_second_degree=ENABLE_2ND, use_residuals=USE_RESIDUALS,
        model_type=MODEL_TYPE, seed=SELECTED_SEED, df=current_df_wide,
        cat_labels=shared['cat_labels_dict'],
        train_end_idx=shared['global_val_start_idx'], save_embeddings=False,
    )
    emb = np.asarray(graph_embeddings, dtype=float)
    cols = pd.to_datetime(current_df_wide.columns)
    end_dates = [cols[min(i + WINDOW_SIZE - 1, len(cols) - 1)] for i in range(emb.shape[0])]
    return emb, pd.to_datetime(end_dates)


_SEASON_BY_MONTH = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Fall', 10: 'Fall', 11: 'Fall'}
_EVENT_FLAGS = ['is_christmas', 'is_christmas_eve', 'is_black_friday',
                'is_thanksgiving', 'is_new_year_eve', 'is_holiday']


def _date_labels(dates, df_p):
    """Season / macro-event / holiday-binary label per window end-date, sourced
    from the selected product's exogenous flag columns."""
    flag_cols = [c for c in _EVENT_FLAGS if c in df_p.columns]
    fmap = {}
    if flag_cols:
        tmp = df_p[[DATE_COL] + flag_cols].copy()
        tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL]).dt.normalize()
        for _, r in tmp.iterrows():
            fmap[r[DATE_COL]] = {c: bool(r[c]) for c in flag_cols}
    seasons, events, holiday = [], [], []
    for d in pd.to_datetime(dates):
        dn = pd.Timestamp(d).normalize()
        seasons.append(_SEASON_BY_MONTH[dn.month])
        f = fmap.get(dn, {})
        if f.get('is_christmas') or f.get('is_christmas_eve'):
            ev = 'Christmas'
        elif f.get('is_black_friday'):
            ev = 'Black Friday'
        elif f.get('is_thanksgiving'):
            ev = 'Thanksgiving'
        elif f.get('is_new_year_eve'):
            ev = 'New Year'
        elif f.get('is_holiday'):
            ev = 'Holiday'
        else:
            ev = 'Normal'
        events.append(ev)
        holiday.append('Holiday' if f.get('is_holiday') else 'Non-Holiday')
    return dict(season=np.array(seasons), event=np.array(events),
                holiday=np.array(holiday))


# ── Analysis 2: Latent-space clustering (t-SNE / UMAP) ────────────────────────
def _reduce_2d(emb, method, seed):
    n = emb.shape[0]
    if method == 'umap':
        import umap  # optional dependency
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(emb), 'UMAP'
    from sklearn.manifold import TSNE
    perp = max(5, min(30, (n - 1) // 3))
    xy = TSNE(n_components=2, perplexity=perp, init='pca',
              learning_rate='auto', random_state=seed).fit_transform(emb)
    return xy, 't-SNE'


def tsne_analysis(emb, labels, dates, out_dir):
    print(f"\n{'='*72}\n# ANALYSIS 2 — Latent-space clustering of Graph2Vec embeddings\n{'='*72}")
    if emb.shape[0] < 5:
        print(f"  Only {emb.shape[0]} windows — too few for a meaningful 2D embedding; skipping.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    event_order  = ['Normal', 'Holiday', 'Thanksgiving', 'Black Friday', 'Christmas', 'New Year']

    def _scatter(ax, xy, lab, title, order):
        cats = [c for c in order if c in set(lab)] or sorted(set(lab))
        cmap = plt.get_cmap('tab10')
        for i, c in enumerate(cats):
            m = lab == c
            ax.scatter(xy[m, 0], xy[m, 1], s=12, alpha=0.7,
                       color=cmap(i % 10), label=c, edgecolors='none')
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=1.6, fontsize=8, loc='best')

    for method in DR_METHODS:
        try:
            xy, name = _reduce_2d(emb, method, SELECTED_SEED)
        except Exception as e:
            print(f"  [{method}] skipped ({type(e).__name__}: {e})")
            continue
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        _scatter(axes[0], xy, labels['season'], f"{name} — by Season", season_order)
        _scatter(axes[1], xy, labels['event'], f"{name} — by Macro-Event", event_order)
        fig.suptitle(f"Graph2Vec latent space ({name}) — metric={METRIC} | seed {SELECTED_SEED}",
                     fontsize=13)
        fig.tight_layout()
        out_pdf = os.path.join(out_dir, f"latent_space_{method}.pdf")
        fig.savefig(out_pdf, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved {name} latent-space plot -> {out_pdf}")

        coords = pd.DataFrame({
            'date': pd.to_datetime(dates), 'x': xy[:, 0], 'y': xy[:, 1],
            'season': labels['season'], 'event': labels['event'], 'holiday': labels['holiday'],
        })
        coords_csv = os.path.join(out_dir, f"latent_space_{method}_coords.csv")
        coords.to_csv(coords_csv, index=False)
        print(f"  Saved {name} coordinates -> {coords_csv}")


# ── Analysis 3: Temporal stability (cosine similarity) ───────────────────────
def cosine_analysis(emb, dates, labels, df_p, out_dir):
    print(f"\n{'='*72}\n# ANALYSIS 3 — Temporal stability cos(z_t, z_t+1)\n{'='*72}")
    if emb.shape[0] < 3:
        print(f"  Only {emb.shape[0]} windows — too few for a cosine series; skipping.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = emb[:-1], emb[1:]
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.where(den > 0, (a * b).sum(1) / np.maximum(den, 1e-12), np.nan)
    d = pd.to_datetime(dates[1:])          # similarity attributed to the later window
    ev = labels['event'][1:]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(d, cos, lw=1.0, color='steelblue', label='cos(z_t, z_{t+1})')
    ax.axhline(np.nanmean(cos), color='gray', ls='--', lw=1,
               label=f'mean={np.nanmean(cos):.3f}')

    event_colors = {'Christmas': 'red', 'Black Friday': 'darkorange',
                    'Thanksgiving': 'purple', 'New Year': 'brown'}
    seen = set()
    for di, e in zip(d, ev):
        if e in event_colors:
            ax.axvline(di, color=event_colors[e], ls=':', lw=1, alpha=0.6,
                       label=e if e not in seen else None)
            seen.add(e)
    promo_cols = [c for c in df_p.columns if c.startswith('promo')]
    if promo_cols:
        pmask = (df_p[promo_cols] == 1).any(axis=1)
        pset = set(pd.to_datetime(df_p.loc[pmask, DATE_COL]).dt.normalize())
        first = True
        for di in d:
            if pd.Timestamp(di).normalize() in pset:
                ax.axvline(di, color='green', ls='-', lw=0.6, alpha=0.25,
                           label='Promo' if first else None)
                first = False

    ax.set_xlabel('window end date'); ax.set_ylabel('cosine similarity')
    ax.set_title(f"Temporal stability of Graph2Vec embeddings — metric={METRIC} | seed {SELECTED_SEED}")
    ax.set_ylim(min(0.0, float(np.nanmin(cos))), 1.01)
    ax.legend(fontsize=8, ncol=2, loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_pdf = os.path.join(out_dir, "temporal_stability_cosine.pdf")
    fig.savefig(out_pdf, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved cosine-stability plot -> {out_pdf}")

    out = pd.DataFrame({'date': d, 'cosine_sim_prev': cos,
                        'event': ev, 'holiday': labels['holiday'][1:]})
    out_csv = os.path.join(out_dir, "temporal_stability_cosine.csv")
    out.to_csv(out_csv, index=False)
    print(f"  Saved cosine series -> {out_csv}")

    hol = labels['holiday'][1:] == 'Holiday'
    if hol.any() and (~hol).any():
        print(f"  mean cos — Non-Holiday: {np.nanmean(cos[~hol]):.4f} | "
              f"Holiday: {np.nanmean(cos[hol]):.4f}")


# ── Driver ───────────────────────────────────────────────────────────────────
def run_one(shared, product_id, store_id, seed):
    """Run the full representation-quality analysis for one (product, seed).

    ``SELECTED_SEED`` / ``SELECTED_PRODUCT`` are set as module globals before this
    call (see ``main``) so the seed-dependent helpers use the current pair.
    """
    prod = prepare_product(shared, product_id, store_id)
    if prod is None:
        print(f"  [skip] product {(product_id, store_id)} has insufficient history.")
        return

    out_dir = os.path.join(OUT_BASE, f"product_{product_id}_{store_id}", f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'#'*72}\nAnalysing product={product_id} store={store_id} seed={seed}\n"
          f"Outputs -> {out_dir}\n{'#'*72}")

    if RUN_ABLATION:
        run_ablation_analysis(shared, prod, product_id, store_id, out_dir)

    if RUN_TSNE or RUN_COSINE:
        emb, dates = extract_graph2vec_embeddings(shared, product_id)
        print(f"\nExtracted {emb.shape[0]} Graph2Vec windows of dim "
              f"{emb.shape[1] if emb.ndim > 1 else 0}.")
        labels = _date_labels(dates, prod['df_p'])
        if RUN_TSNE:
            tsne_analysis(emb, labels, dates, out_dir)
        if RUN_COSINE:
            cosine_analysis(emb, dates, labels, prod['df_p'], out_dir)


def resolve_pairs(shared):
    """The (item_id, store_id, seed) triples to analyse.

    Prefers SELECTED_PRODUCT_SEED_PAIRS (item_id, seed) with store = STORE_ID,
    de-duplicated while preserving order; otherwise falls back to the single
    SELECTED_PRODUCT / SELECTED_SEED pair.
    """
    if SELECTED_PRODUCT_SEED_PAIRS:
        triples, seen = [], set()
        for item_id, seed in SELECTED_PRODUCT_SEED_PAIRS:
            key = (int(item_id), int(STORE_ID), int(seed))
            if key not in seen:
                seen.add(key)
                triples.append(key)
        return triples
    pid, sid = _resolve_selected_product(shared)
    return [(int(pid), int(sid), int(SELECTED_SEED))]


def main():
    global SELECTED_SEED, SELECTED_PRODUCT
    shared = prepare_shared()
    triples = resolve_pairs(shared)
    print(f"Running representation-quality analysis on {len(triples)} (product, seed) pair(s).")

    for i, (product_id, store_id, seed) in enumerate(triples, 1):
        # Mutate the globals the seed-dependent helpers read, so each pair uses
        # its own seed/product without threading the seed through every function.
        SELECTED_SEED = seed
        SELECTED_PRODUCT = (product_id, store_id)
        print(f"\n[{i}/{len(triples)}] product={product_id} store={store_id} seed={seed}")
        try:
            run_one(shared, product_id, store_id, seed)
        except Exception as e:
            print(f"  [error] pair (product={product_id}, seed={seed}) failed: "
                  f"{type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
