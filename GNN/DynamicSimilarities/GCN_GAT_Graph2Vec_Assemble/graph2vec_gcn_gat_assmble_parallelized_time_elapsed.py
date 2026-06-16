import os
import sys
import pickle
import itertools
import time
import csv as _csv_mod
import csv
import hashlib
import random
import subprocess
import threading
import glob as _glob_mod
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Paths & sys.path setup ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)                                          # local modules (inference, train, ...)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))     # DynamicSimilarities/ (plots, utils)

# Shared modules from DynamicSimilarities/
from plots import plot_results
from utils import (
    generate_exogenous_features, compute_metrics, neighbourhood_graph,
    compute_distances_1vsAll, compute_similarities_1vsAll, _save_node_features_csv
)

# Local (per-step) GCN/GAT modules.  The forward signature
# (pyg_batch, target_node_indices, ts_seq) -> (B, horizon, 1) is identical
# across all four variants, so a single inference / train / dataset serves them.
from node_feature_builders import (
    node_feature_width as _nf_width,
    resolve_feature_names,
)
# Paired dataset loaders — one per temporal head, mirroring the train_methods
# and inference_strategies pairing.  Both expose an identical positional
# signature (target_data, exog_data, lookback, pyg_graphs, graph_window_size)
# and an identical collate, so the runner stays head-agnostic via dispatch.
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
from inference_strategies import _recursive_forecast_gcn_mlp_perstep, _recursive_forecast_gcn_lstm_perstep
# Graph2Vec variants: frozen-embedding forecasters that ride the per-step
# pipeline by carrying each day's embedding as a 1-node graph feature.
from generate_graph2vecwithadaptativethreshold import load_or_generate_embeddings
from graph2vec_assemble import (
    recursive_forecast_graph2vec_perstep,
    embedding_to_single_node_data,
)
from ablationlstm import AblationLSTMForecaster
from ablationmlp import AblationMLPForecaster
from train_methods import train_gcn_lstm_model , train_gcn_mlpmodel


from LSTMBaseline.dataset import TimeSeriesDataset as LSTMBaselineDataset
from LSTMBaseline.lstm import LSTM as LSTMBaseline
from LSTMBaseline.lstm_train import train_model as train_lstm_baseline
from LSTMBaseline.lstm_inference import recursive_inference as recursive_forecast_lstm_baseline

from MLP_Baseline.train import train_mlp_forecaster, TrainConfig
from MLP_Baseline.inference import recursive_inference_dynamic_exog
from MLP_Baseline.utils import ExogenousScaler

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
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/data_andre_classified.feather'))
TOP_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/top_12500.feather'))
DATE_COL = 'date'
TARGET_COL = 'value'


val_size = 61
forecast_horizon = 153
train_size = 761 - val_size - forecast_horizon
lookback_window = 30
BATCH_SIZE = 32


# Leave as None to run all (item_id, store_id) pairs found in DATA_PATH.
# Override with a list of tuples to run only a subset, e.g.:
#PRODUCTS_TO_TEST = [(26008, 6269)]
PRODUCTS_TO_TEST = None
EXOG_COLS_LSTM = [
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
EXOG_COLS_MLP = [
    # Cyclical Calendar Features 
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "dom_sin", "dom_cos", "wom_sin", "wom_cos", "month_sin", "month_cos", "quarter_sin", "quarter_cos", "woy_sin", "woy_cos",
 
    # Structural boundaries 
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
   
    # Trend Hint 
    "rolling_mean_excl_7",
   
 
    # Holidays & Events (Crucial)
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_bridge_day"
]
# Union of all exog columns — used by generate_exogenous_features so every
# feature is computed once; each variant subsets to its own list at runtime.
EXOG_COLS = list(dict.fromkeys(EXOG_COLS_LSTM + EXOG_COLS_MLP))
grid_configs = [
    {'metric': 'spearman', 'thresholds': [0.634]},
    #{'metric': 'cid', 'thresholds': [4.17]},
]

window_sizes              = [30]
step_sizes                = [1]
enable_edges_opts         = [True]
enable_second_degree_opts = [False]
USE_RESIDUALS  = False
MODEL_TYPE     = 'ridge'
EPOCHS         = 1000
PATIENCE       = 100
# Head-specific learning rates / dropout — the LSTM and MLP heads (and their
# matching baselines) overfit at different rates, so they are tuned separately.
LEARNING_RATE_LSTM = 0.001   # gcn_lstm / gat_lstm / lstm ablation + LSTM baseline
LEARNING_RATE_MLP  = 0.0001   # gcn_mlp  / gat_mlp  / mlp  ablation + MLP  baseline
HIDDEN_SIZE    = 32
NUM_LAYERS     = 1
DROPOUT_LSTM   = 0.0          # LSTM-head variants + LSTM baseline
DROPOUT_MLP    = 0.2          # MLP-head variants + MLP baseline
D_G            = 16          # per-step graph embedding dim
SAVE_MODELS         = False
SAVE_PLOTS          = True
RUN_LSTM_BASELINE   = True
RUN_MLP_BASELINE    = True
USE_EMBEDDINGS = True
SAVE_EMBEDDINGS = False
# Node feature mode for the GCN ego-graphs (must match across dataset build,
# inference patching and the ablation dummy-graph width):
#   'raw'      full window sequence as node features (width = window_size)
#   'stats'    8-dim statistical summary per node
#   'catch22'  22 catch22 shape/dynamics features (scale-invariant)
#   'catch24'  22 catch22 + DN_Mean + DN_Spread_Std (24-d; restores scale)
#   'catch24_minmax'  catch24 + min + max (26-d; explicit combo, no zero_fraction)
#   'catch24_minmaxlast'  catch24 + min + max + last (27-d; adds window's last value)
# Add new combinations in node_feature_builders._MODE_FEATURE_LISTS — every
# call site resolves through resolve_feature_names, so widths stay consistent.
node_feature_modes = ['catch24_minmaxlast']

def _node_feature_width(mode, window_size, extra_stats=True):
    """Node-feature dim for a given mode — used to size ablation dummy graphs.

    Resolves through the same ``resolve_feature_names`` used by the dataset
    build and inference patch, so the ablation width can never drift from the
    real feature layout.
    """
    names = resolve_feature_names(mode, extra_stats=extra_stats)
    return _nf_width(names, window_size=window_size)


# ── Model-architecture grid ────────────────────────────────────────────────
# The four assembled variants share the same per-step forward signature
# (pyg_batch, target_node_indices, ts_seq) -> (B, horizon, 1) and the same
# dataset / collate / inference, so the ONLY thing that changes per variant is
# the class constructed by build_forecaster().  Ablation (z=0) removes the
# graph branch entirely, collapsing GCN==GAT, so it is run once per head
# (lstm/mlp) via the dedicated graph-free models — see the dedup in the grid.
MODEL_VARIANTS   = ['graph2vec_lstm', 'graph2vec_mlp', 'gcn_lstm', 'gcn_mlp', 'gat_lstm', 'gat_mlp']
ATTENTION_HEADS  = 4            # GAT attention heads (encoder layer 1)
MLP_HIDDEN_SIZES = (128,64)     # TimeDistributed-MLP head widths


def _variant_head(variant):
    """'lstm' or 'mlp' — the temporal head implied by a variant name."""
    return 'mlp' if variant.endswith('mlp') else 'lstm'


def build_forecaster(variant, ablate_z, in_channels, ts_input_size):
    """
    Construct the forecaster for ``variant``.

    When ``ablate_z`` is True the graph branch is dropped entirely by using the
    dedicated graph-free model for the head type (GCN vs GAT is irrelevant once
    z is gone), giving a fair no-graph parameter count.  All returned models
    accept the same forward signature, so the caller is variant-agnostic.
    """
    head = _variant_head(variant)
    # LSTM and MLP heads are regularised separately (see globals).
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
            ts_input_size=ts_input_size, hidden_sizes=MLP_HIDDEN_SIZES, lookback_window=lookback_window,
             horizon=1, dropout=dropout,
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

# ── Diagnostic switches ───────────────────────────────────────────────────
# Grid of ablation modes to run in a single execution:
#   True  — GCN z-embedding zeroed (ablation)
#   False — full GCN+LSTM model
ABLATE_Z_VALUES = [True, False]
# Seeds for independent runs — each seed fully controls torch/numpy/random init.

SEEDS = [42, 1000, 26008, 555555,213626, 907969, 5219788, 13451285, 23616558, 618626816]
# Per-epoch diagnostics (||z||, Var(z), ‖∇gcn‖/‖∇lstm‖) are appended to
# this CSV.  Suffix "_ablation" is added automatically when ABLATE_Z=True.
DIAG_CSV_NAME  = "diagnostics.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Parallelism config ──────────────────────────────────────────────────
# PARALLEL_MODE: "product" → one subprocess per product-shard (recommended)
#                "seed"    → one subprocess per seed (each handles all products)
# MAX_CONCURRENT caps simultaneous processes (Ryzen 7 5600X: 6 physical cores).
# N_PRODUCT_WORKERS: how many product shards to create (product mode only).
# PRODUCTS_END: only distribute PRODUCTS_TO_TEST[:PRODUCTS_END]; None = all.
#               Use to leave the last product(s) for a sequential run later.
# NO_MERGE=True skips merging the per-worker CSVs into the canonical {metric}.csv.
PARALLEL = True
PARALLEL_MODE = "product"   # "seed" or "product"
MAX_CONCURRENT = 6
N_PRODUCT_WORKERS = 6       # product mode: products / N_PRODUCT_WORKERS each
PRODUCTS_END = 60           # exclude PRODUCTS_TO_TEST[60:] from the parallel run
NO_MERGE = False


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
# Parallel launchers (one subprocess per product-shard or per seed)
# ──────────────────────────────────────────────────────────────────────────
def _merge_worker_csvs(metrics):
    """Merge per-worker ``{metric}_w*.csv`` files into the canonical ``{metric}.csv``."""
    for metric in metrics:
        partial_files = sorted(_glob_mod.glob(os.path.join(SCRIPT_DIR, f"{metric}_w*.csv")))
        if not partial_files:
            continue
        seen = set()
        rows = []
        header = None
        for p in partial_files:
            try:
                with open(p, newline="") as f:
                    reader = csv.DictReader(f)
                    header = reader.fieldnames or header
                    for row in reader:
                        key = (
                            row.get("product_id", ""), row.get("store_id", ""),
                            row.get("seed", ""), row.get("metric", ""),
                            row.get("model_variant", ""),
                            row.get("threshold", ""), row.get("ablate_z", ""),
                            row.get("node_feature_mode", ""),
                        )
                        if key not in seen:
                            seen.add(key)
                            rows.append(row)
            except Exception as e:
                print(f"  Warning: could not read {p}: {e}")
        if header is None:
            continue
        out_path = os.path.join(SCRIPT_DIR, f"{metric}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Merged {len(rows)} rows from {len(partial_files)} files -> {out_path}")


def _spawn_workers(jobs, max_concurrent, metrics, no_merge=False):
    """
    Launch one subprocess per (worker_id, num_workers, extra_env) job, bounded
    to ``max_concurrent`` at a time, then optionally merge per-worker CSVs.
    """
    semaphore = threading.Semaphore(max_concurrent)
    results = {}
    threads = []

    def _run(wid, nw, extra_env, log_name):
        with semaphore:
            env = {
                **os.environ,
                "GCN_WORKER_ID": str(wid),
                "GCN_NUM_WORKERS": str(nw),
                # 1 thread each so the processes share cores cleanly instead of
                # all fighting for every core.
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                **extra_env,
            }
            cmd = [sys.executable, __file__]
            log_path = os.path.join(SCRIPT_DIR, log_name)
            print(f"[START] worker={wid}/{nw}  log={log_name}", flush=True)
            t0 = time.time()
            with open(log_path, "w") as log_fh:
                proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                                        text=True, env=env)
                pid = proc.pid
                print(f"[RUN]   worker={wid} pid={pid}", flush=True)
                try:
                    rc = proc.wait()
                except KeyboardInterrupt:
                    proc.terminate()
                    proc.wait()
                    raise
            elapsed = time.time() - t0
            status = "OK" if rc == 0 else f"FAILED(rc={rc})"
            print(f"[DONE]  worker={wid} pid={pid}  {status}  {elapsed/60:.1f} min", flush=True)
            results[wid] = rc

    print(f"Spawning {len(jobs)} workers  |  max_concurrent={max_concurrent}")
    for wid, nw, extra_env, log_name in jobs:
        t = threading.Thread(target=_run, args=(wid, nw, extra_env, log_name), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    failed = [w for w, rc in results.items() if rc != 0]
    print(f"\nCompleted {len(results)} workers.  Failed: {len(failed)}")
    for w in failed:
        print(f"  worker={w}  (check its log file)")

    if not no_merge:
        print(f"\nMerging partial CSVs -> {{metric}}.csv")
        _merge_worker_csvs(metrics)


def _run_parallel_products(num_workers, max_concurrent, metrics, no_merge=False):
    """One subprocess per disjoint product slice; each runs all SEEDS for its shard."""
    jobs = [
        (wid, num_workers, {}, f"log_worker{wid}_of{num_workers}.txt")
        for wid in range(num_workers)
    ]
    _spawn_workers(jobs, max_concurrent, metrics, no_merge=no_merge)


def _run_parallel_seeds(seeds, max_concurrent, metrics, no_merge=False):
    """One subprocess per seed; each handles all products for that single seed."""
    jobs = [
        (wid, 1, {"GCN_SEED": str(seed)}, f"log_seed{seed}.txt")
        for wid, seed in enumerate(seeds)
    ]
    _spawn_workers(jobs, max_concurrent, metrics, no_merge=no_merge)


# ──────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────
def main():
    worker_id   = int(os.environ.get("GCN_WORKER_ID", "0"))
    num_workers = int(os.environ.get("GCN_NUM_WORKERS", "1"))

    # Limit PyTorch's intra-op thread pool to 1 when running as a product-shard
    # worker so the processes share the cores instead of all claiming them.
    if num_workers > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # ── Parallel launcher (coordinator only) ─────────────────────────────
    # Workers are identified by GCN_NUM_WORKERS > 1 (product mode) or by
    # GCN_SEED being set (seed mode).  The bare coordinator has neither.
    is_worker = num_workers > 1 or "GCN_SEED" in os.environ
    if PARALLEL and not is_worker:
        metrics = [c['metric'] for c in grid_configs]
        if PARALLEL_MODE == "seed":
            _run_parallel_seeds(
                seeds=SEEDS, max_concurrent=MAX_CONCURRENT,
                metrics=metrics, no_merge=NO_MERGE,
            )
        else:  # "product"
            _run_parallel_products(
                num_workers=N_PRODUCT_WORKERS, max_concurrent=MAX_CONCURRENT,
                metrics=metrics, no_merge=NO_MERGE,
            )
        return

    # Seed-mode workers run a single seed; product-mode workers run all SEEDS.
    seeds_to_run = ([int(os.environ["GCN_SEED"])] if "GCN_SEED" in os.environ
                    else list(SEEDS))

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
        print(f"Loading top products from {TOP_DATA_PATH}...")
        top_df = pd.read_feather(TOP_DATA_PATH)
        PRODUCTS_TO_TEST = (
            top_df[["item_id", "store_id"]]
            .drop_duplicates()
            .sort_values(["item_id", "store_id"])
            .apply(lambda r: (int(r["item_id"]), int(r["store_id"])), axis=1)
            .tolist()
        )
        print(f"Running all {len(PRODUCTS_TO_TEST)} (item_id, store_id) pairs from dataset.")

    # Apply PRODUCTS_END so the last product(s) can be excluded from the
    # parallel run and handled sequentially later.
    if PRODUCTS_END is not None:
        PRODUCTS_TO_TEST = PRODUCTS_TO_TEST[:PRODUCTS_END]

    # ── Product sharding (parallelism across workers) ────────────────────
    # Each worker processes a disjoint slice so multiple processes run
    # simultaneously without duplicating work.  In seed mode (num_workers==1)
    # this is a no-op and every worker handles all products for its seed.
    n_products_total = len(PRODUCTS_TO_TEST)
    PRODUCTS_TO_TEST = PRODUCTS_TO_TEST[worker_id::num_workers]
    print(f"Worker {worker_id}/{num_workers}: processing "
          f"{len(PRODUCTS_TO_TEST)} / {n_products_total} products  |  seeds={seeds_to_run}")

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

    # ── Per-worker CSV suffix (avoids concurrent-write races) ────────────
    # Each worker writes to its own {metric}_w{id}.csv; the coordinator merges
    # them into the canonical {metric}.csv after all workers finish.
    worker_csv_suffix = f"_w{worker_id}" if num_workers > 1 else ""

    # ── Timing bookkeeping ───────────────────────────────────────────────
    timings_csv_path = os.path.join(SCRIPT_DIR, f"timings{worker_csv_suffix}.csv")
    product_times = []          # (product_id, store_id, seed, seconds)
    total_t0 = time.time()

    # ── Per-product graph cache ──────────────────────────────────────────
    # The neighbourhood-graph pipeline (similarity scan + PyG build + timeline
    # alignment + train/val/seed/future slicing) depends only on
    # (metric, threshold/pct, window, step, enable_edges, enable_second_degree,
    # node_feature_mode) for a given product — NOT on the seed or the model
    # variant.  Build each distinct config once per product and reuse it across
    # all seeds and all four variants.  Seeds for a product are processed
    # consecutively, so resetting on product change keeps memory bounded.
    _graph_cache = {}
    # Graph2Vec embeddings are seed-dependent, so this cache is keyed including
    # the seed (graph2vec_lstm/_mlp of one seed share); reset per product to
    # bound memory, mirroring _graph_cache.
    _graph2vec_cache = {}
    _graph_cache_product = None

    # ── Progress tracking (this worker's shard only) ─────────────────────
    # Workers are independent subprocesses over disjoint product slices, so
    # PRODUCTS_TO_TEST here is already this worker's share. Each (product,
    # store, seed) is one work unit; we report done/remaining + ETA per worker
    # into this worker's own log file.
    total_units = len(PRODUCTS_TO_TEST) * len(seeds_to_run)
    units_done = 0

    for (product_id, store_id), seed in itertools.product(PRODUCTS_TO_TEST, seeds_to_run):
        if (product_id, store_id) != _graph_cache_product:
            _graph_cache = {}
            _graph2vec_cache = {}
            _graph_cache_product = (product_id, store_id)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"\n{'='*80}")
        print(f"PROCESSING PRODUCT {product_id} FOR STORE {store_id}  [Seed {seed}]")
        print(f"{'='*80}\n")

        df_p = (
            full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)]
            .sort_values(DATE_COL).reset_index(drop=True)
        )

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

        # Fit one scaler per head so each variant uses its own column set.
        exog_scalers_by_head = {}
        exog_train_by_head   = {}
        exog_val_by_head     = {}
        exog_test_by_head    = {}
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
        # Aliases used by the LSTM baseline (always lstm head).
        exog_train_scaled = exog_train_by_head['lstm']
        exog_val_scaled   = exog_val_by_head['lstm']
        exog_test_scaled  = exog_test_by_head['lstm']
        exog_scaler       = exog_scalers_by_head['lstm']

        product_t0 = time.time()

        best_models_dir_base  = os.path.join(SCRIPT_DIR, 'best_models')
        os.makedirs(best_models_dir_base,  exist_ok=True)

        # GCN+LSTM requires a graph for every config — no_emb baseline is dropped.
        all_configs = list(grid_configs)

        # ── LSTM Baseline (run once per product) ──────────────────────────
        _bl_forecast = None
        _bl_t_losses: list = []
        _bl_v_losses: list = []
        _bl_rmse = _bl_mae = _bl_bias = _bl_score = _bl_pocid = None
        if RUN_LSTM_BASELINE:
            print(f"\n{'='*60}")
            print("Running LSTM Baseline...")
            print(f"{'='*60}")
            lstm_input_bl = 1 + (len(EXOG_COLS_LSTM) if EXOG_COLS_LSTM else 0)
            # Re-seed so this model's init is independent of anything that drew
            # from the shared RNG stream earlier in this product/seed iteration.
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            bl_model = LSTMBaseline(
                input_size=lstm_input_bl,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT_LSTM,
            ).to(device)
            bl_train_ds = LSTMBaselineDataset(train_scaled, exog_train_scaled if EXOG_COLS_LSTM else None, lookback_window)
            bl_val_ds   = LSTMBaselineDataset(val_scaled,   exog_val_scaled   if EXOG_COLS_LSTM else None, lookback_window)
            bl_train_loader = DataLoader(bl_train_ds, batch_size=BATCH_SIZE, shuffle=False)
            bl_val_loader   = DataLoader(bl_val_ds,   batch_size=BATCH_SIZE, shuffle=False)
            bl_optimizer = torch.optim.AdamW(bl_model.parameters(), lr=LEARNING_RATE_LSTM)
            bl_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                bl_optimizer, mode='min', factor=0.5, patience=PATIENCE // 3,
            )
            bl_model_path = os.path.join(
                best_models_dir_base,
                f"lstm_baseline_{product_id}_{store_id}_exp{seed}.pth",
            )
            if os.path.exists(bl_model_path):
                print(f"Loading existing LSTM baseline from {bl_model_path}...")
                bl_model.load_state_dict(torch.load(bl_model_path, map_location=device))
            else:
                bl_model, _bl_t_losses, _bl_v_losses, _, _ = train_lstm_baseline(
                    epochs=EPOCHS, model=bl_model,
                    train_loader=bl_train_loader, val_loader=bl_val_loader,
                    exog_cols=EXOG_COLS_LSTM,
                    criterion=nn.MSELoss(), criterion2=nn.MSELoss(),
                    optimizer=bl_optimizer, device=device,
                    best_model_path=bl_model_path,
                    scheduler=bl_scheduler, patience=PATIENCE,
                )
                bl_model.load_state_dict(torch.load(bl_model_path, map_location=device))

            exog_test_raw = df_p[EXOG_COLS_LSTM][test_slice].values if EXOG_COLS_LSTM else None
            _bl_forecast, _ = recursive_forecast_lstm_baseline(
                model=bl_model,
                test_start_idx=test_start_idx,
                seq_length=lookback_window,
                val_scaled=val_scaled,
                exog_val_scaled=exog_val_scaled if EXOG_COLS_LSTM else None,
                exog_test_scaled=exog_test_scaled if EXOG_COLS_LSTM else None,
                exog_test=exog_test_raw,
                scaler=scaler,
                exog_scaler=exog_scaler if EXOG_COLS_LSTM else None,
                df_product=df_p,
                device=device,
                exog_cols=EXOG_COLS_LSTM,
                forecast_window=forecast_horizon,
                strategy='best_val',
                item_id=product_id,
                store_id=store_id,
                loss_type='MSELoss',
                script_dir=SCRIPT_DIR,
            )
            _bl_arr   = np.array(_bl_forecast, dtype=float)
            _bl_valid = ~np.isnan(_bl_arr)
            if _bl_valid.any():
                # Use the same compute_metrics as the GCN path so score (the
                # weighted composite, not r2) and pocid are directly comparable.
                _bl_rmse, _bl_mae, _bl_bias, _bl_score, _bl_pocid = compute_metrics(
                    test[_bl_valid], _bl_arr[_bl_valid]
                )
                _bl_rmse  = float(_bl_rmse)
                _bl_mae   = float(_bl_mae)
                _bl_bias  = float(_bl_bias)
                _bl_score = float(_bl_score)
                _bl_pocid = float(_bl_pocid)
                print(f"LSTM Baseline RMSE: {_bl_rmse:.4f}")

        # ── MLP Baseline (run once per product) ───────────────────────────
        _mlp_forecast = None
        _mlp_t_losses: list = []
        _mlp_v_losses: list = []
        _mlp_rmse = _mlp_mae = _mlp_bias = _mlp_score = _mlp_pocid = None
        if RUN_MLP_BASELINE:
            print(f"\n{'='*60}")
            print("Running MLP Baseline...")
            print(f"{'='*60}")
            # Type-aware exog scaler (pass-through for binary/cyclical, MinMax
            # for continuous) — required by the MLP_Baseline train/inference API.
            mlp_exog_scaler = ExogenousScaler(continuous_strategy='minmax')
            mlp_exog_scaler.fit(df_p[EXOG_COLS_MLP].iloc[train_slice], EXOG_COLS_MLP)

            # Re-seed so the MLP baseline's init is independent of the LSTM
            # baseline (which drew from the same RNG stream just above).
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            mlp_cfg = TrainConfig(
                lookback=lookback_window,
                horizon=1,
                batch_size=BATCH_SIZE,
                train_size=train_size,
                val_size=val_size,
                lr=LEARNING_RATE_MLP,
                dropout=DROPOUT_MLP,
                epochs=EPOCHS,
                patience=PATIENCE,
                hidden_sizes=MLP_HIDDEN_SIZES,
                device=str(device),
            )
            mlp_model, _, _mlp_t_losses, _mlp_v_losses, _ = train_mlp_forecaster(
                df=df_p, cfg=mlp_cfg, seed=seed, loss_type='mse',
                product_id=f"{product_id}_{store_id}", scaler=scaler, target_channel=0,
                target_col=TARGET_COL, exog_cols=EXOG_COLS_MLP, test_size=forecast_horizon,
                exog_scaler=mlp_exog_scaler,
            )

            # Leak-safe recursive inference: lag/rolling cols are recomputed from
            # the running prediction buffer, never from ground-truth test. Full
            # train+val target history is passed so long lags resolve correctly.
            _mlp_hist_target = np.concatenate([train, val]).astype(np.float32)
            _mlp_hist_exog   = df_p[EXOG_COLS_MLP].iloc[val_slice].iloc[-lookback_window:].reset_index(drop=True)
            _mlp_future_exog = df_p[EXOG_COLS_MLP].iloc[test_slice].reset_index(drop=True)

            _mlp_forecast = recursive_inference_dynamic_exog(
                model=mlp_model,
                target_scaler=scaler,
                exog_scaler=mlp_exog_scaler,
                exog_cols=EXOG_COLS_MLP,
                history_target_unscaled=_mlp_hist_target,
                history_exog_unscaled=_mlp_hist_exog,
                future_exog_unscaled=_mlp_future_exog,
                target_channel=0,
                device=str(device),
            )
            _mlp_arr   = np.array(_mlp_forecast, dtype=float)
            _mlp_valid = ~np.isnan(_mlp_arr)
            if _mlp_valid.any():
                # Same compute_metrics as the GCN/LSTM paths so score (the weighted
                # composite, not r2) and pocid are directly comparable.
                _mlp_rmse, _mlp_mae, _mlp_bias, _mlp_score, _mlp_pocid = compute_metrics(
                    test[_mlp_valid], _mlp_arr[_mlp_valid]
                )
                _mlp_rmse  = float(_mlp_rmse)
                _mlp_mae   = float(_mlp_mae)
                _mlp_bias  = float(_mlp_bias)
                _mlp_score = float(_mlp_score)
                _mlp_pocid = float(_mlp_pocid)
                print(f"MLP Baseline RMSE: {_mlp_rmse:.4f}")

        for config in all_configs:
            metric      = config['metric']
            thresholds  = config.get('thresholds',  [None])
            percentiles = config.get('percentiles', [None])

            results_by_w_s = {}
            _neighbour_series: dict = {}
            _all_step_neighbours: dict = {}  # step_idx -> neighbour IDs (every forecast step)

            is_threshold_mode = thresholds is not None and thresholds != [None]
            params   = thresholds if is_threshold_mode else percentiles
            
            grid_search_plots_dir = os.path.join(SCRIPT_DIR, 'grid_search_plots', f'seed_{seed}', f'product_{product_id}_store_{store_id}')
            os.makedirs(grid_search_plots_dir, exist_ok=True)

            # ── Persist baseline results to the per-metric CSV ───────────
            # Baselines are graph-free, so all graph/threshold columns are left
            # blank.  Written once per (product, seed, metric), same schema as
            # the variant rows below, so everything lives in one CSV.
            _baseline_rows = []
            if _bl_forecast is not None:
                _baseline_rows.append(
                    ("lstm_baseline", _bl_rmse, _bl_mae, _bl_bias, _bl_score, _bl_pocid))
            if _mlp_forecast is not None:
                _baseline_rows.append(
                    ("mlp_baseline", _mlp_rmse, _mlp_mae, _mlp_bias, _mlp_score, _mlp_pocid))
            if _baseline_rows:
                baseline_csv_path = os.path.join(SCRIPT_DIR, f"{metric}{worker_csv_suffix}.csv")
                _bl_file_exists = os.path.exists(baseline_csv_path)
                with open(baseline_csv_path, 'a', newline='') as _bl_csvfile:
                    _bl_writer = csv.writer(_bl_csvfile)
                    if not _bl_file_exists:
                        _bl_writer.writerow([
                            "product_id", "store_id", "seed", "metric", "model_variant",
                            "window_size", "step_size", "threshold", "percentile",
                            "enable_edges", "enable_second_degree", "ablate_z",
                            "node_feature_mode",
                            "rmse", "mae", "bias", "r2_score", "pocid",
                        ])
                    for _mv, _r, _m, _b, _s, _p in _baseline_rows:
                        _bl_writer.writerow([
                            product_id, store_id, seed, metric, _mv,
                            "", "", "", "", "", "", "", "",
                            _r, _m, _b, _s, _p,
                        ])

            iterator = itertools.product(
                MODEL_VARIANTS, ABLATE_Z_VALUES, params, window_sizes, step_sizes,
                enable_edges_opts, enable_second_degree_opts, node_feature_modes,
            )

            for (model_variant, ablate_z, param_val, window_size, step_size,
                 enable_edges, enable_second_degree, node_feature_mode) in iterator:
                # Ablation drops the graph branch, so GCN/GAT/Graph2Vec all
                # collapse to the same graph-free model.  Run ablation once per
                # head (via the gcn_* variants) and skip the redundant gat_* and
                # graph2vec_* ablations.
                if ablate_z and (model_variant.startswith('gat')
                                 or model_variant.startswith('graph2vec')):
                    continue
                # When ablating, z is zeroed so the threshold has no effect on
                # the trained weights.  Skip all but the first threshold value.
                if ablate_z and param_val != params[0]:
                    continue
                if ablate_z and node_feature_mode != node_feature_modes[0]:
                    continue

                # Per-variant exog columns and pre-scaled data.
                _head         = _variant_head(model_variant)
                is_graph2vec  = model_variant.startswith('graph2vec')
                exog_cols     = EXOG_COLS_LSTM if _head == 'lstm' else EXOG_COLS_MLP
                exog_train_scaled = exog_train_by_head[_head]
                exog_val_scaled   = exog_val_by_head[_head]
                exog_test_scaled  = exog_test_by_head[_head]
                exog_scaler       = exog_scalers_by_head[_head]

                # Per-head dataset loader / collate / graph-builder (mirrors the
                # train_fn and inference dispatch).  Both datasets share an
                # identical positional signature and collate output.
                if _head == 'lstm':
                    DatasetClass    = GCN_LSTMTimeSeriesDataset
                    collate_fn      = collate_pyg_ts_lstm
                    build_graphs_fn = build_pyg_graphs_lstm
                else:
                    DatasetClass    = GCNMLPTimeSeriesDataset
                    collate_fn      = collate_pyg_ts_mlp
                    build_graphs_fn = build_pyg_graphs_mlp

                current_threshold  = param_val if is_threshold_mode else None
                current_percentile = param_val if not is_threshold_mode else None

                key = (model_variant, ablate_z, param_val, window_size, step_size)
                if key not in results_by_w_s:
                    results_by_w_s[key] = {
                        'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                        'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {},
                        'threshold': None,
                    }

                print(f"\n{'='*60}")
                param_str = (f"threshold={current_threshold}" if is_threshold_mode
                             else f"percentile={current_percentile}")
                print(f"Running Experiment: variant={model_variant}, ablate_z={ablate_z}, "
                      f"metric={metric}, {param_str}, "
                      f"window_size={window_size}, enable_edges={enable_edges}, "
                      f"2nd_degree={enable_second_degree}, node_features={node_feature_mode}")
                print(f"{'='*60}")

                # ── 1 & 2. Graph pipeline (skipped for ablation) ─────────
                metric_type = infer_metric_type(metric)
                # Graph2Vec inference needs the fitted embedding model; GCN/GAT
                # leave it None.  current_df_wide / product_offset are also set
                # by the graph2vec branch for the dynamic re-inference step.
                graph2vec_model = None
                if ablate_z:
                    # GCN output is zeroed in forward(); building hundreds of
                    # sliding-window graphs would be pure waste.  Pass dummy
                    # single-node placeholder graphs instead.
                    fixed_threshold = None
                    results_by_w_s[key]['threshold'] = fixed_threshold
                    _dummy = Data(
                        x=torch.zeros(1, _node_feature_width(node_feature_mode, window_size),
                                      dtype=torch.float32),
                        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                        edge_attr=torch.zeros(1, 1, dtype=torch.float32),
                        num_nodes=1,
                    )
                    pyg_train         = [_dummy] * (val_start_idx  - train_start_idx)
                    pyg_val           = [_dummy] * (test_start_idx - val_start_idx)
                    pyg_seed_graphs   = [_dummy] * lookback_window
                    pyg_future_graphs = [_dummy] * forecast_horizon
                elif is_graph2vec:
                    # ── Graph2Vec branch ─────────────────────────────────────
                    # One frozen embedding per sliding window, carried as a
                    # 1-node graph so the per-step GCN dataset/collate/deque are
                    # reused verbatim.  Embeddings are seed-dependent (Doc2Vec
                    # seed) so the cache key includes the seed; graph2vec_lstm and
                    # graph2vec_mlp of the same seed share the embeddings.
                    distance_metrics = ['euclidean', 'manhattan', 'hamming', 'amplitude_offset',
                                        'slope_consistency', 'phase_invariance', 'dtw', 'cid',
                                        'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
                    current_df_wide = df_wide_scaled if metric in distance_metrics else df_wide_global
                    T_global       = current_df_wide.shape[1]
                    product_offset = T_global - len(df_p)

                    g2v_key = (metric, param_val, window_size, step_size,
                               enable_edges, enable_second_degree, seed)
                    cached = _graph2vec_cache.get(g2v_key)
                    if cached is None:
                        (graph_embeddings, graph2vec_model, _g2v_csv,
                         _g2v_bt, _g2v_et, fixed_threshold) = load_or_generate_embeddings(
                            product_id=product_id,
                            metric=metric,
                            metric_type=metric_type,
                            window_size=window_size,
                            step_size=step_size,
                            threshold=current_threshold if is_threshold_mode else None,
                            percentile=current_percentile if not is_threshold_mode else None,
                            dimensions=D_G,
                            enable_edges_within_star=enable_edges,
                            enable_second_degree=enable_second_degree,
                            use_residuals=USE_RESIDUALS,
                            model_type=MODEL_TYPE,
                            seed=seed,
                            df=current_df_wide,
                            cat_labels=cat_labels_dict,
                            train_end_idx=global_val_start_idx,
                            save_embeddings=False,
                        )
                        print(f"Resolved graph2vec threshold={current_threshold}: {fixed_threshold}")

                        # one 1-node graph per sliding window (embedding = node feature)
                        pyg_windows = [
                            embedding_to_single_node_data(graph_embeddings[i], target_label=product_id)
                            for i in range(len(graph_embeddings))
                        ]
                        pyg_aligned_global = _align_pyg_windows_to_timeline(
                            pyg_windows, window_size=window_size,
                            step_size=step_size, T=T_global,
                        )
                        pyg_train = pyg_aligned_global[product_offset + train_start_idx:
                                                       product_offset + val_start_idx]
                        pyg_val   = pyg_aligned_global[product_offset + val_start_idx:
                                                       product_offset + test_start_idx]
                        seed_start = product_offset + test_start_idx - lookback_window
                        seed_end   = product_offset + test_start_idx
                        pyg_seed_graphs = pyg_aligned_global[seed_start:seed_end]
                        # Future graphs are recomputed dynamically at inference
                        # (from the model's own predictions); kept only for API
                        # symmetry with the GCN path — the graph2vec inference
                        # ignores them.
                        fut_start = product_offset + test_start_idx
                        fut_end   = fut_start + forecast_horizon
                        pyg_future_graphs = pyg_aligned_global[fut_start:fut_end]

                        _graph2vec_cache[g2v_key] = {
                            'fixed_threshold':   fixed_threshold,
                            'graph2vec_model':   graph2vec_model,
                            'pyg_train':         pyg_train,
                            'pyg_val':           pyg_val,
                            'pyg_seed_graphs':   pyg_seed_graphs,
                            'pyg_future_graphs': pyg_future_graphs,
                        }
                        cached = _graph2vec_cache[g2v_key]
                    else:
                        print(f"Reusing cached graph2vec embeddings for {g2v_key}")

                    fixed_threshold   = cached['fixed_threshold']
                    graph2vec_model   = cached['graph2vec_model']
                    pyg_train         = cached['pyg_train']
                    pyg_val           = cached['pyg_val']
                    pyg_seed_graphs   = cached['pyg_seed_graphs']
                    pyg_future_graphs = cached['pyg_future_graphs']
                    results_by_w_s[key]['threshold'] = fixed_threshold
                else:
                    # ── 1. Resolve wide matrix / compute func (cheap, per-iter) ──
                    distance_metrics = ['euclidean', 'manhattan', 'hamming', 'amplitude_offset',
                                        'slope_consistency', 'phase_invariance', 'dtw', 'cid',
                                        'lorentzian', 'sbd', 'msm', 'edr', 'lcss']
                    current_df_wide = df_wide_scaled if metric in distance_metrics else df_wide_global
                    compute_func = (compute_distances_1vsAll if metric_type == 'distance'
                                    else compute_similarities_1vsAll)
                    T_global       = current_df_wide.shape[1]
                    product_offset = T_global - len(df_p)

                    # ── 2. Build (or reuse) seed-/variant-independent graphs ──
                    # The key holds everything the graph pipeline depends on; it
                    # deliberately excludes the seed and the model variant, so the
                    # first non-ablation variant of the first seed builds the
                    # graphs and every later seed × variant reuses them.
                    graph_key = (metric, param_val, window_size, step_size,
                                 enable_edges, enable_second_degree, node_feature_mode)
                    cached = _graph_cache.get(graph_key)
                    if cached is None:
                        # ── Build per-window NX graphs ───────────────────
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

                        # ── NX -> per-window PyG, align to timeline (per-day) ──
                        pyg_windows = build_graphs_fn(
                            nx_graphs, current_df_wide, product_id,
                            window_size=window_size, step_size=step_size,
                            node_feature_mode=node_feature_mode,
                        )
                        _save_node_features_csv(
                            pyg_windows, product_id, store_id, metric,
                            fixed_threshold, window_size, node_feature_mode, SCRIPT_DIR,
                        )
                        pyg_aligned_global = _align_pyg_windows_to_timeline(
                            pyg_windows, window_size=window_size,
                            step_size=step_size, T=T_global,
                        )
                        pyg_train = pyg_aligned_global[product_offset + train_start_idx:
                                                       product_offset + val_start_idx]
                        pyg_val   = pyg_aligned_global[product_offset + val_start_idx:
                                                       product_offset + test_start_idx]
                        # Inference seed: the L per-day graphs ending at the last validation day.
                        seed_start = product_offset + test_start_idx - lookback_window
                        seed_end   = product_offset + test_start_idx
                        pyg_seed_graphs = pyg_aligned_global[seed_start:seed_end]
                        # Future graphs aligned to each forecast day t in [test_start, test_start+H)
                        fut_start = product_offset + test_start_idx
                        fut_end   = fut_start + forecast_horizon
                        pyg_future_graphs = pyg_aligned_global[fut_start:fut_end]

                        _graph_cache[graph_key] = {
                            'fixed_threshold':   fixed_threshold,
                            'pyg_train':         pyg_train,
                            'pyg_val':           pyg_val,
                            'pyg_seed_graphs':   pyg_seed_graphs,
                            'pyg_future_graphs': pyg_future_graphs,
                        }
                        cached = _graph_cache[graph_key]
                    else:
                        print(f"Reusing cached graphs for {graph_key}")

                    fixed_threshold   = cached['fixed_threshold']
                    pyg_train         = cached['pyg_train']
                    pyg_val           = cached['pyg_val']
                    pyg_seed_graphs   = cached['pyg_seed_graphs']
                    pyg_future_graphs = cached['pyg_future_graphs']
                    results_by_w_s[key]['threshold'] = fixed_threshold

                # ── 3. Datasets / loaders (PER-STEP) ─────────────────────
                # Positional construction: both heads share the signature
                # (target_data, exog_data, lookback, pyg_graphs, graph_window_size).
                # LSTM names the 3rd arg seq_length, MLP names it lookback_window.
                use_pin_memory = torch.cuda.is_available()
                train_dataset = DatasetClass(
                    train_scaled,
                    exog_train_scaled if exog_cols else None,
                    lookback_window,
                    pyg_train,
                    graph_window_size=window_size,
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                    pin_memory=use_pin_memory, collate_fn=collate_fn,
                )
                val_dataset = DatasetClass(
                    val_scaled,
                    exog_val_scaled if exog_cols else None,
                    lookback_window,
                    pyg_val,
                    graph_window_size=window_size,
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                    pin_memory=use_pin_memory, collate_fn=collate_fn,
                )

                # ── 4. Model + optimiser ─────────────────────────────────
                in_channels     = pyg_train[0].x.shape[1]    # depends on node_feature_mode (e.g. 24 for catch24)
                lstm_input_size = 1 + (len(exog_cols) if exog_cols else 0)
                # Re-seed so each variant's init is independent of the models
                # built before it in this iteration (decouples LSTM/MLP variants
                # so changing one head never perturbs the other's results).
                torch.manual_seed(seed)
                np.random.seed(seed % (2**32))
                random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                model = build_forecaster(
                    model_variant, ablate_z, in_channels, lstm_input_size,
                ).to(device)
                model.ablate_z = ablate_z
                criterion  = nn.MSELoss()
                criterion2 = nn.MSELoss()
                lr_for_head = LEARNING_RATE_LSTM if _head == 'lstm' else LEARNING_RATE_MLP
                optimizer  = torch.optim.AdamW(model.parameters(), lr=lr_for_head)
                scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=PATIENCE // 3,
                )

                # ── 5. Checkpoint paths (PER-STEP tag) ───────────────────
                model_dir_label = (f"th{current_threshold}" if is_threshold_mode
                                   else f"pct{current_percentile}")
                best_models_dir = os.path.join(
                    best_models_dir_base, str(window_size), str(step_size),
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
                    f"best_{model_variant}_perstep_{prefix_star}{product_id}_{metric}"
                    f"_w{window_size}_s{step_size}_{param_label}_nf{node_feature_mode}{res_tag}_exp{seed}"
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
                        DIAG_CSV_NAME.replace(
                            ".csv", f"_{model_variant}{diag_suffix}_exp{seed}.csv"
                        ),
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
                    train_fn = (train_gcn_lstm_model if _variant_head(model_variant) == 'lstm'
                                else train_gcn_mlpmodel)
                    model, train_losses, val_losses, best_epoch, train_time = train_fn(
                        seed=seed,
                        epochs=EPOCHS, model=model,
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

                # Build position->k map for the lag columns present in exog_cols
                lag_col_indices = {
                    exog_cols.index(name): k
                    for name, k in LAG_K_BY_NAME.items()
                    if name in exog_cols
                }
                # Build position->W map for rolling_mean_excl_W columns
                rolling_mean_excl_col_indices = {}
                for i, name in enumerate(exog_cols):
                    if name.startswith(ROLLING_MEAN_EXCL_PREFIX):
                        try:
                            W = int(name[len(ROLLING_MEAN_EXCL_PREFIX):])
                        except ValueError:
                            continue
                        rolling_mean_excl_col_indices[i] = W
                # Raw history that ends at the day BEFORE the first forecast step
                target_history_unscaled = np.concatenate([train, val]).astype(np.float32)

                # Extract target-node initial window in the same scale as current_df_wide
                # so that the per-step recursive forecaster can patch future_graphs[i].x[0]
                # with the model's own rolling predictions (avoids test-period leakage).
                initial_target_window = None
                target_z_scaler = None
                if not ablate_z and product_id in current_df_wide.index:
                    _tw_start = max(0, product_offset + test_start_idx - window_size)
                    initial_target_window = current_df_wide.loc[product_id].values[
                        _tw_start: product_offset + test_start_idx
                    ].astype(np.float32)
                    if metric_type == 'distance' and product_id in product_scalers:
                        # predictions come out of MinMaxScaler; convert to z-score
                        # to match df_wide_scaled space used for distance metrics.
                        target_z_scaler = product_scalers[product_id]

                if is_graph2vec:
                    # Graph2Vec inference rebuilds the dynamic graph from the
                    # model's own rolling prediction each step and re-infers the
                    # embedding (the frozen pre-built future graphs would leak the
                    # target's actual future values).  Head-agnostic: the LSTM and
                    # MLP graph2vec models share the forward contract.
                    forecast = recursive_forecast_graph2vec_perstep(
                        model=model,
                        ts_seed=ts_seed,
                        initial_graphs=pyg_seed_graphs,
                        exog_test_scaled=exog_test_scaled if exog_cols else None,
                        scaler=scaler,
                        horizon=forecast_horizon,
                        device=device,
                        graph2vec_model=graph2vec_model,
                        df_wide=current_df_wide,
                        cat_labels=cat_labels_dict,
                        target_id=product_id,
                        metric=metric,
                        fixed_threshold=fixed_threshold,
                        graph_window_size=window_size,
                        first_forecast_col=product_offset + test_start_idx,
                        target_seed_window_dfscale=initial_target_window,
                        enable_edges_within_star=enable_edges,
                        enable_second_degree=enable_second_degree,
                        target_df_scaler=target_z_scaler,
                        target_history_unscaled=target_history_unscaled if exog_cols else None,
                        lag_col_indices=lag_col_indices if exog_cols else None,
                        rolling_mean_excl_col_indices=rolling_mean_excl_col_indices if exog_cols else None,
                        exog_scaler=exog_scaler if exog_cols else None,
                    )
                elif _variant_head(model_variant) == 'lstm':
                    forecast = _recursive_forecast_gcn_lstm_perstep(
                        model=model,
                        ts_seed=ts_seed,
                        initial_graphs=pyg_seed_graphs,
                        future_graphs=pyg_future_graphs,
                        exog_test_scaled=exog_test_scaled if exog_cols else None,
                        scaler=scaler,
                        horizon=forecast_horizon,
                        device=device,
                        target_history_unscaled=target_history_unscaled if exog_cols else None,
                        lag_col_indices=lag_col_indices if exog_cols else None,
                        rolling_mean_excl_col_indices=rolling_mean_excl_col_indices if exog_cols else None,
                        exog_scaler=exog_scaler if exog_cols else None,
                        initial_target_window=initial_target_window,
                        target_z_scaler=target_z_scaler,
                        node_feature_mode=node_feature_mode,
                    )
                else:
                    forecast = _recursive_forecast_gcn_mlp_perstep(
                        model=model,
                        ts_seed=ts_seed,
                        initial_graphs=pyg_seed_graphs,
                        future_graphs=pyg_future_graphs,
                        exog_test_scaled=exog_test_scaled if exog_cols else None,
                        scaler=scaler,
                        horizon=forecast_horizon,
                        device=device,
                        target_history_unscaled=target_history_unscaled if exog_cols else None,
                        lag_col_indices=lag_col_indices if exog_cols else None,
                        rolling_mean_excl_col_indices=rolling_mean_excl_col_indices if exog_cols else None,
                        exog_scaler=exog_scaler if exog_cols else None,
                        exog_cols=exog_cols if exog_cols else None,
                        node_feature_mode=node_feature_mode,
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
                label_name = (f"{model_variant}|{param_str_label}|w:{window_size}|st:{step_size}"
                              f"|e:{enable_edges}|2nd:{enable_second_degree}|nf:{node_feature_mode}|az:{az_str}")

                results_by_w_s[key]['forecasts'][label_name]    = forecast
                results_by_w_s[key]['train_losses'][label_name] = train_losses
                results_by_w_s[key]['val_losses'][label_name]   = val_losses
                results_by_w_s[key]['rmse'][label_name]         = rmse
                results_by_w_s[key]['mae'][label_name]          = mae
                results_by_w_s[key]['bias'][label_name]         = bias
                results_by_w_s[key]['score'][label_name]        = score
                results_by_w_s[key]['pocid'][label_name]        = pocid

                print(f"Finished {metric} @ {param_val} -> RMSE: {rmse}\n")

                # ── Append to persistent CSV (per metric, per worker) ────
                csv_results_path = os.path.join(SCRIPT_DIR, f"{metric}{worker_csv_suffix}.csv")
                file_exists = os.path.exists(csv_results_path)
                with open(csv_results_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow([
                            "product_id", "store_id", "seed", "metric", "model_variant",
                            "window_size", "step_size", "threshold", "percentile",
                            "enable_edges", "enable_second_degree", "ablate_z",
                            "node_feature_mode",
                            "rmse", "mae", "bias", "r2_score", "pocid",
                        ])
                    writer.writerow([
                        product_id, store_id, seed, metric, model_variant,
                        window_size, step_size,
                        current_threshold if current_threshold is not None else "",
                        current_percentile if current_percentile is not None else "",
                        enable_edges, enable_second_degree, ablate_z,
                        node_feature_mode,
                        rmse, mae, bias, score, pocid,
                    ])

            # ── Per-metric combined plot (all thresholds) ────────────────
            train_index = df_p[DATE_COL][train_slice].values
            val_index   = df_p[DATE_COL][val_slice].values
            test_index  = df_p[DATE_COL][test_slice].values

            # Group every (variant, ablate_z, threshold) result by (w, s) so a
            # single overlay shows all four variants + their ablations together.
            # Each label_name already carries the variant, so labels stay unique.
            grouped_results = {}
            for (mv, az, pv, w, s), res_dicts in results_by_w_s.items():
                gkey = (w, s)
                if gkey not in grouped_results:
                    grouped_results[gkey] = {
                        'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                        'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {},
                    }
                for k in grouped_results[gkey]:
                    grouped_results[gkey][k].update(res_dicts[k])

            # ── Merged overlay: all variants + ablations on the same plot ────
            if SAVE_PLOTS:
                for (w, s), src in grouped_results.items():
                    if not src['forecasts']:
                        continue

                    raw_str = ("_".join(map(str, thresholds))
                               if thresholds is not None and len(thresholds) > 0 and percentiles is None
                               else "_".join(map(str, percentiles)))
                    values_str = hashlib.md5(raw_str.encode()).hexdigest()[:8]
                    sub_dir = os.path.join(
                        grid_search_plots_dir, metric_type, f'window_{w}', f'step_{s}',
                        f'item_{product_id}', values_str, f'exp_{seed}',
                    )
                    os.makedirs(sub_dir, exist_ok=True)

                    # All variant/ablation labels are already disambiguated by
                    # label_name, so copy the grouped dict straight through.
                    merged = {k: dict(v) for k, v in src.items()}

                    # Add the LSTM baseline as a final overlay line
                    if _bl_forecast is not None:
                        merged['forecasts']['LSTM Baseline']    = _bl_forecast
                        merged['train_losses']['LSTM Baseline'] = _bl_t_losses
                        merged['val_losses']['LSTM Baseline']   = _bl_v_losses
                        merged['rmse']['LSTM Baseline']         = _bl_rmse
                        merged['mae']['LSTM Baseline']          = _bl_mae
                        merged['bias']['LSTM Baseline']         = _bl_bias
                        merged['score']['LSTM Baseline']        = _bl_score
                        merged['pocid']['LSTM Baseline']        = _bl_pocid

                    # Add the MLP baseline as a final overlay line
                    if _mlp_forecast is not None:
                        merged['forecasts']['MLP Baseline']    = _mlp_forecast
                        merged['train_losses']['MLP Baseline'] = _mlp_t_losses
                        merged['val_losses']['MLP Baseline']   = _mlp_v_losses
                        merged['rmse']['MLP Baseline']         = _mlp_rmse
                        merged['mae']['MLP Baseline']          = _mlp_mae
                        merged['bias']['MLP Baseline']         = _mlp_bias
                        merged['score']['MLP Baseline']        = _mlp_score
                        merged['pocid']['MLP Baseline']        = _mlp_pocid

                    merged_path = os.path.join(
                        sub_dir,
                        f"item_{product_id}_{metric}_all_configs_merged.html",
                    )
                    print(f"Saving merged overlay plot to: {os.path.abspath(merged_path)}")
                    plot_results(
                        train, val, test, merged['forecasts'],
                        train_index, val_index, test_index,
                        merged['train_losses'], merged['val_losses'],
                        metric=metric, embedding_strategy='gcn_perstep',
                        window_size=w, step_size=s, threshold=None, percentile=None,
                        target_col=TARGET_COL,
                        title=(f'GCN/GAT × LSTM/MLP vs Ablation (z=0) — '
                               f'{metric} | W={w} | S={s} | Item={product_id}'),
                        save_path=merged_path,
                        rmse=merged['rmse'], mae=merged['mae'],
                        bias=merged['bias'], score=merged['score'],
                        pocid=merged['pocid'],
                        neighbour_series=_neighbour_series if _neighbour_series else None,
                        all_step_neighbours=_all_step_neighbours if _all_step_neighbours else None,
                    )

        # ── Per-product timing ─────────────────────────────────────────────
        product_elapsed = time.time() - product_t0
        product_times.append((product_id, store_id, seed, product_elapsed))
        print(f"\n[TIMING] Product {product_id} | store {store_id} | exp {seed}: "
              f"{product_elapsed:.1f} s ({product_elapsed/60:.2f} min)")

        # ── Progress + ETA (this worker's shard) ───────────────────────────
        # ETA uses the average wall-clock per completed unit (which folds in
        # graph-build overhead), extrapolated over the units still remaining.
        units_done += 1
        units_left = total_units - units_done
        run_elapsed = time.time() - total_t0
        avg_per_unit = run_elapsed / units_done
        eta_seconds = avg_per_unit * units_left
        print(f"[PROGRESS] Worker {worker_id}/{num_workers}: "
              f"{units_done}/{total_units} done | {units_left} remaining "
              f"| elapsed {run_elapsed/60:.1f} min | avg {avg_per_unit/60:.2f} min/unit "
              f"| ETA {eta_seconds/60:.1f} min ({eta_seconds/3600:.2f} h)")

    # ── Total timing & persist ────────────────────────────────────────────────
    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    print(f"  TOTAL: {total_elapsed:.1f} s  ({total_elapsed/60:.2f} min)  "
          f"across {len(PRODUCTS_TO_TEST)} products")

    with open(timings_csv_path, "w", newline="") as fh:
        w = _csv_mod.writer(fh)
        w.writerow(["product_id", "store_id", "seed", "seconds"])
        for pid, sid, eidx, sec in product_times:
            w.writerow([pid, sid, eidx, f"{sec:.3f}"])
        w.writerow(["TOTAL_ALL", "", "", f"{total_elapsed:.3f}"])
    print(f"  Timings written to: {timings_csv_path}")



if __name__ == '__main__':
    main()
