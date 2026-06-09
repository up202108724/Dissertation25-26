import os
import random
import sys
import pickle
import itertools
import hashlib
import csv
import time
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
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '../..')))  # DynamicSimilarities/

from utils import generate_exogenous_features, compute_metrics, neighbourhood_graph, compute_distances_1vsAll, compute_similarities_1vsAll # GraphAnalysis/utils.py

# Reused per-step GCN pipeline (model-agnostic)
from gcn_tdmlpdataset import (
    GCNTimeSeriesDataset,
    collate_pyg_ts,
    build_pyg_graphs_from_nx_windows,
)
from gcn_mlpinference import _recursive_forecast_gcn_perstep

# Local MLP-headed model + training loop + plotting
from gcn_mlp_model import SimpleGCNMLPForecaster
from gcn_mlp_train import train_model
from ablationmlp import AblationMLPForecaster
from plots import plot_results, plot_networkx_plotly


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


# ── Constants ──────────────────────────────────────────────────────────────
FULL_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../../dataset/data_andre_classified.feather'))
TOP_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../../dataset/top_12500.feather'))
DATE_COL = 'date'
TARGET_COL = 'value'

val_size = 30  # FIXED: Must match graph2vec baseline
forecast_horizon = 153
train_size = 761 - val_size - forecast_horizon  # 455
lookback_window = 30

EXOG_COLS = [
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
grid_configs = [

    {'metric': 'spearman', 'thresholds': [0.70]},
]

# Training hyperparameters
BATCH_SIZE = 32
HIDDEN_SIZES = (128, 64)
GCN_HIDDEN_SIZE = 64
MLP_MODEL_TYPE = "tdmlp"
DROPOUT = 0.2
EPOCHS = 1000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MODEL_TYPE = 'ridge'
LOSS_TYPE = 'mse'
PATIENCE = 150  # FIXED: LSTM baseline uses 100, now harmonized to 150
##########################
#SEEDS = [42]
SEEDS = [42, 1000, 26008, 555555,213626, 907969, 5219788, 13451285, 23616558, 618626816]
#SEEDS = [42, 1000, 26008, 555555,213626, 907969]
WINDOW_SIZES = [30]     
STEP_SIZES = [1]
ENABLE_EDGES_OPTS = [True]
ENABLE_SECOND_DEGREE_OPTS = [False]  # We will keep this False for the main analysis, but you can set to True to include second-degree neighbors in the graph construction
USE_RESIDUALS = False
SAVE_PLOTS = True
USE_EMBEDDINGS = True
SAVE_EMBEDDINGS = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_EMBEDDINGS_DIM = 16

# Grid of ablation modes — True: pure TDMLP (no GCN); False: full GCN+MLP.
ABLATE_Z_VALUES = [True,False]  # [True, False]  -- ablations disabled
# Hidden sizes for the ablation-only TDMLP — kept small for a fair no-graph baseline.
ABLATION_HIDDEN_SIZES = (128, 64)
DIAG_CSV_NAME  = "diagnostics.csv"
# Set True to emit inference_graph_log.csv with per-step neighbourhood data.
RECORD_INFERENCE_GRAPHS = False
SAVE_INFERENCE_GRAPHS_PLOTS = False

# ── Parallelism config ────────────────────────────────────────────────────
# PARALLEL_MODE: "seed"    → one subprocess per seed (original behaviour)
#                "product" → one subprocess per product-shard (new)
# MAX_CONCURRENT caps simultaneous processes (Ryzen 7 5600X: 6 physical cores).
# N_PRODUCT_WORKERS: how many product shards to create (product mode only).
# PRODUCTS_END: only distribute products[:PRODUCTS_END]; set to None for all.
#               Use to leave the last product for a sequential run later.
# NO_MERGE=True skips merging the per-worker CSVs into a single output file.
PARALLEL = True
PARALLEL_MODE = "product"   # "seed" or "product"
MAX_CONCURRENT = 6
N_PRODUCT_WORKERS = 6       # product mode: 60 products / 6 workers = 10 each
PRODUCTS_END = 60           # exclude products[60:] from the parallel run
NO_MERGE = False

# Node feature modes for GCN graphs — swept as an extra grid axis so every mode
# is evaluated under the same splits/seeds within a single run.
# 'raw'     — full window sequence as node features (shape: n_nodes × window_size)
# 'stats'   — 8-dim statistical summary per node (mean, std, min, max, first, last, slope, sum)
# 'catch22' — 22 catch22 shape/dynamics features (scale-invariant)
# 'catch24' — 22 catch22 + DN_Mean + DN_Spread_Std (24-d; restores scale)
node_feature_modes = ['catch24', 'stats', 'raw']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Per-day alignment helper (identical to LSTM sibling) ───────────────────
def _make_pad_graph(template: Data) -> Data:
    in_feats = template.x.shape[1]
    return Data(
        x=torch.zeros(1, in_feats, dtype=torch.float32),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        edge_attr=torch.zeros(1, 1, dtype=torch.float32),
        num_nodes=1,
    )


def _align_pyg_windows_to_timeline(pyg_windows, window_size, step_size, T):
    if step_size != 1:
        raise NotImplementedError("Per-day alignment helper currently assumes step_size=1")
    pad = _make_pad_graph(pyg_windows[0])
    aligned = [pad] * window_size + list(pyg_windows)
    if len(aligned) < T:
        aligned += [aligned[-1]] * (T - len(aligned))
    else:
        aligned = aligned[:T]
    return aligned


# Exog columns whose value at day t is value[t-k] — recomputed at inference
# from the rolling (own-prediction-augmented) target history.
LAG_PREFIX = "lag_"
ROLLING_MEAN_EXCL_PREFIX = "rolling_mean_excl_"


# ──────────────────────────────────────────────────────────────────────────
# Seed-parallel launcher (one subprocess per seed, bounded by --max-concurrent)
# ──────────────────────────────────────────────────────────────────────────

_CSV_HEADER = [
    "item_id", "store_id", "seed", "metric",
    "window_size", "step_size", "threshold", "percentile",
    "enable_edges", "enable_second_degree", "ablate_z",
    "node_feature_mode",
    "rmse", "mae", "bias", "r2_score", "pocid",
    "train_time_s", "inference_time_s",
]


def _run_seed_job(seed, worker_id, num_workers):
    env = {
        **os.environ,
        "GCN_SEED": str(seed),
        "GCN_WORKER_ID": str(worker_id),
        "GCN_NUM_WORKERS": str(num_workers),
    }
    cmd = [sys.executable, __file__]
    log_path = os.path.join(SCRIPT_DIR, f"log_seed{seed}_w{worker_id}.txt")
    print(f"[START] seed={seed} w={worker_id}/{num_workers}  log={os.path.basename(log_path)}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
        pid = proc.pid
        print(f"[RUN]   seed={seed} pid={pid}", flush=True)
        rc = proc.wait()
    elapsed = time.time() - t0
    status = "OK" if rc == 0 else f"FAILED(rc={rc})"
    print(f"[DONE]  seed={seed} pid={pid}  {status}  {elapsed/60:.1f} min", flush=True)
    return rc


def _merge_seed_csvs(output_path):
    partial_files = sorted(_glob_mod.glob(os.path.join(SCRIPT_DIR, "gcn_mlp_results_seed*.csv")))
    if not partial_files:
        print("No partial CSVs found to merge.")
        return
    seen = set()
    rows = []
    for p in partial_files:
        try:
            with open(p, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("item_id", "").startswith("TOTAL"):
                        continue
                    key = (
                        row.get("item_id", ""), row.get("store_id", ""),
                        row.get("seed", ""), row.get("metric", ""),
                        row.get("threshold", ""), row.get("ablate_z", ""),
                        row.get("node_feature_mode", ""),
                    )
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
        except Exception as e:
            print(f"  Warning: could not read {p}: {e}")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(rows)} rows from {len(partial_files)} files -> {output_path}")


def _run_parallel_seeds(seeds, max_concurrent, num_workers_per_seed=1, no_merge=False):
    semaphore = threading.Semaphore(max_concurrent)
    results = {}
    threads = []

    def _run(seed, wid, nw):
        with semaphore:
            rc = _run_seed_job(seed, wid, nw)
            results[(seed, wid)] = rc

    print(f"Spawning {len(seeds) * num_workers_per_seed} jobs  |  max_concurrent={max_concurrent}")
    print(f"Seeds: {seeds}  |  product_shards_per_seed={num_workers_per_seed}\n")
    for seed in seeds:
        for wid in range(num_workers_per_seed):
            t = threading.Thread(target=_run, args=(seed, wid, num_workers_per_seed), daemon=True)
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    failed = [(s, w) for (s, w), rc in results.items() if rc != 0]
    print(f"\nCompleted {len(results)} jobs.  Failed: {len(failed)}")
    if failed:
        for s, w in failed:
            print(f"  seed={s} worker={w}  (see log_seed{s}_w{w}.txt)")

    if not no_merge:
        merged_csv = os.path.join(SCRIPT_DIR, "gcn_mlp_results.csv")
        print(f"\nMerging partial CSVs -> {merged_csv}")
        _merge_seed_csvs(merged_csv)


def _run_parallel_products(num_workers, max_concurrent, no_merge=False):
    """
    Spawn ``num_workers`` subprocesses, each handling a disjoint product slice
    via GCN_WORKER_ID / GCN_NUM_WORKERS env vars.  Bounded to ``max_concurrent``
    at a time.  Each worker runs all SEEDS sequentially for its product shard.
    """
    semaphore = threading.Semaphore(max_concurrent)
    results = {}
    threads = []

    def _run(wid, nw):
        with semaphore:
            env = {
                **os.environ,
                "GCN_WORKER_ID": str(wid),
                "GCN_NUM_WORKERS": str(nw),
                # Each worker gets 1 thread so N_PRODUCT_WORKERS processes
                # share the cores cleanly instead of all fighting for all of them.
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            cmd = [sys.executable, __file__]
            log_path = os.path.join(SCRIPT_DIR, f"log_worker{wid}_of{nw}.txt")
            print(f"[START] worker={wid}/{nw}  log={os.path.basename(log_path)}", flush=True)
            t0 = time.time()
            with open(log_path, "w") as log_fh:
                proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env)
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

    print(f"Spawning {num_workers} product-shard workers  |  max_concurrent={max_concurrent}")
    for wid in range(num_workers):
        t = threading.Thread(target=_run, args=(wid, num_workers), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    failed = [w for w, rc in results.items() if rc != 0]
    print(f"\nCompleted {len(results)} workers.  Failed: {len(failed)}")
    if failed:
        for w in failed:
            print(f"  worker={w}  (see log_worker{w}_of{num_workers}.txt)")

    if not no_merge:
        merged_csv = os.path.join(SCRIPT_DIR, "gcn_mlp_results.csv")
        print(f"\nMerging partial CSVs -> {merged_csv}")
        _merge_seed_csvs(merged_csv)


# ──────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────
def main():
    worker_id   = int(os.environ.get("GCN_WORKER_ID", "0"))
    num_workers = int(os.environ.get("GCN_NUM_WORKERS", "1"))

    # Limit PyTorch's intra-op thread pool to 1 when running as a worker so
    # N_PRODUCT_WORKERS processes share the CPU cores instead of all claiming them.
    if num_workers > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # ── Parallel launcher (coordinator only) ──────────────────────────────
    # Workers are identified by GCN_NUM_WORKERS > 1 (product mode) or
    # GCN_SEED being set (seed mode).  The bare coordinator invocation has
    # neither env var set.
    is_worker = num_workers > 1 or "GCN_SEED" in os.environ
    if PARALLEL and not is_worker:
        if PARALLEL_MODE == "seed":
            _run_parallel_seeds(
                seeds=SEEDS,
                max_concurrent=MAX_CONCURRENT,
                num_workers_per_seed=1,
                no_merge=NO_MERGE,
            )
        else:  # "product"
            _run_parallel_products(
                num_workers=N_PRODUCT_WORKERS,
                max_concurrent=MAX_CONCURRENT,
                no_merge=NO_MERGE,
            )
        return

    print(f"Loading data from {FULL_DATA_PATH}...")
    df = pd.read_feather(FULL_DATA_PATH)
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

    cat_labels_dict = (
        full_df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()
        if 'cat_label' in full_df.columns else {}
    )
    df_wide_global = full_df.pivot_table(
        index='item_id', columns=DATE_COL, values=TARGET_COL, aggfunc='sum'
    ).fillna(0)
    df_wide_global.columns = pd.to_datetime(df_wide_global.columns).strftime('%Y-%m-%d')

    
    # Globally define train split based on identical logic for all items
    L = len(df_wide_global.columns)
    global_train_start_idx = L - forecast_horizon - val_size - train_size
    global_val_start_idx = L - forecast_horizon - val_size

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
        
    experience_start_time = time.time()
    
    top_df = pd.read_feather(TOP_DATA_PATH)
    products_df = top_df[['item_id', 'store_id']].drop_duplicates().reset_index(drop=True)
    all_products = list(products_df.itertuples(index=False, name=None))

    # Apply PRODUCTS_END so the last product(s) are excluded from the
    # parallel run and can be handled sequentially later.
    if PRODUCTS_END is not None:
        all_products = all_products[:PRODUCTS_END]

    # ── Product sharding (parallelism across workers) ─────────────────────
    # Each worker processes a disjoint slice of all_products so multiple
    # processes can run simultaneously without duplicating work.
    PRODUCTS_TO_TEST = all_products[worker_id::num_workers]
    print(f"Worker {worker_id}/{num_workers}: processing {len(PRODUCTS_TO_TEST)} / {len(all_products)} products.")

    # ── Per-worker CSV file (avoids concurrent-write races) ───────────────
    # Each worker writes to its own file; run_parallel.py merges them.
    worker_csv_suffix = f"_w{worker_id}" if num_workers > 1 else ""
    seed_tag = f"_seed{SEEDS[0]}" if len(SEEDS) == 1 else ""
    results_csv_name = f"gcn_mlp_results{seed_tag}{worker_csv_suffix}.csv"
    results_csv = os.path.join(SCRIPT_DIR, results_csv_name)

    # Seed done_set from ALL existing result files so resumption works across
    # workers and previous runs.
    done_set = set()
    for _f in os.listdir(SCRIPT_DIR):
        if _f.startswith("gcn_mlp_results") and _f.endswith(".csv") and "backup" not in _f:
            _p = os.path.join(SCRIPT_DIR, _f)
            try:
                _d = pd.read_csv(_p, dtype=str)
                if 'item_id' not in _d.columns:
                    continue
                _d['threshold'] = _d['threshold'].fillna('').astype(str)
                for _, row in _d.iterrows():
                    done_set.add((str(row['item_id']), str(row['store_id']),
                                  str(row['seed']), str(row['metric']),
                                  str(row['threshold']), str(row['ablate_z'])))
            except Exception:
                pass
    print(f"Resuming: {len(done_set)} experiments already completed.")

    for product_id, store_id in PRODUCTS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"PROCESSING PRODUCT {product_id} FOR STORE {store_id}")
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
        test_scaled  = scaler.transform(test.reshape(-1, 1)).flatten()

        # Val context: prepend lookback_window rows so GCNTimeSeriesDataset produces
        # val_size windows instead of (val_size - lookback_window) windows.
        val_ctx_slice  = slice(val_start_idx - lookback_window, test_start_idx)
        val_ctx        = df_p[TARGET_COL][val_ctx_slice].values
        val_scaled_ctx = scaler.transform(val_ctx.reshape(-1, 1)).flatten()

        
        # Type-aware scaler: pass-through for binary/cyclical, MinMax for continuous.
        exog_scaler = ExogenousScaler(continuous_strategy='minmax')
        exog_scaler.fit(df_p[EXOG_COLS].iloc[train_slice], EXOG_COLS)
        exog_train_scaled   = exog_scaler.transform(df_p[EXOG_COLS].iloc[train_slice].copy(), EXOG_COLS).values
        exog_val_scaled     = exog_scaler.transform(df_p[EXOG_COLS].iloc[val_slice].copy(), EXOG_COLS).values
        exog_test_scaled    = exog_scaler.transform(df_p[EXOG_COLS].iloc[test_slice].copy(), EXOG_COLS).values
        exog_val_scaled_ctx = exog_scaler.transform(df_p[EXOG_COLS].iloc[val_ctx_slice].copy(), EXOG_COLS).values
        # Keep UNSCALED copies — needed by the dynamic inference loop.
        exog_train_unscaled = df_p[EXOG_COLS].iloc[train_slice].copy()
        exog_val_unscaled = df_p[EXOG_COLS].iloc[val_slice].copy()
        exog_test_unscaled = df_p[EXOG_COLS].iloc[test_slice].copy()
        for seed in SEEDS:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            print(f"\n--- RUNNING WITH SEED {seed} ---\n")

            grid_search_plots_dir = os.path.join(SCRIPT_DIR, 'grid_search_plots', f'seed_{seed}', f'product_{product_id}_store_{store_id}')
            best_models_seed_dir  = os.path.join(SCRIPT_DIR, 'best_models',       f'seed_{seed}', f'product_{product_id}_store_{store_id}')
            os.makedirs(grid_search_plots_dir, exist_ok=True)
            os.makedirs(best_models_seed_dir,  exist_ok=True)

            all_configs = list(grid_configs)
            ##Baseline execution first
            # Skip if this baseline (product, seed) was already saved in a previous run.
            _bl_done_key = (str(product_id), str(store_id), str(seed), 'N/A', '', 'baseline')
            if _bl_done_key in done_set:
                print(f"Baseline already done for product={product_id}, store={store_id}, seed={seed}. Skipping.")
                _bl_forecast = _bl_t_losses = _bl_v_losses = None
                _bl_rmse = _bl_mae = _bl_bias = _bl_score = _bl_pocid = None
            else:
                # Prepare global config for this strategy
                cfg = TrainConfig(
                    lookback=lookback_window,
                    horizon=1,
                    batch_size=BATCH_SIZE,
                    train_size=train_size,
                    val_size=val_size,
                    lr=LEARNING_RATE,
                    dropout=DROPOUT,
                    epochs=EPOCHS,
                    patience=PATIENCE,
                    hidden_sizes=HIDDEN_SIZES,
                    weight_decay=WEIGHT_DECAY,
                    model_type=MLP_MODEL_TYPE,  # FIXED: Was missing!
                    device=str(DEVICE)
                )

                start_train = time.time()
                model, _, t_losses, v_losses, best_epoch = train_mlp_forecaster(
                    df=df_p, cfg=cfg, seed=seed, loss_type=LOSS_TYPE,
                    product_id=f"{product_id}_{store_id}", scaler=scaler, target_channel=0,
                    target_col=TARGET_COL, exog_cols=EXOG_COLS, test_size=forecast_horizon,
                    exog_scaler=exog_scaler,
                )
                train_time = time.time() - start_train

                # Leak-safe recursive inference: lag/rolling cols are recomputed
                # from the running prediction buffer, never from ground-truth test.
                # Full train+val target passed so lag_364 resolves correctly.
                recent_target_unscaled = np.concatenate([train, val]).astype(np.float32)
                recent_exog_unscaled_df = exog_val_unscaled.iloc[-lookback_window:].reset_index(drop=True)

                start_infer = time.time()
                forecast = recursive_inference_dynamic_exog(
                    model=model,
                    target_scaler=scaler,
                    exog_scaler=exog_scaler,
                    exog_cols=EXOG_COLS,
                    history_target_unscaled=recent_target_unscaled,
                    history_exog_unscaled=recent_exog_unscaled_df,
                    future_exog_unscaled=exog_test_unscaled.reset_index(drop=True),
                    target_channel=0,
                    device=str(device),
                )
                infer_time = time.time() - start_infer

                # Metrics
                rmse = np.sqrt(mean_squared_error(test, forecast))
                mae = mean_absolute_error(test, forecast)
                bias = np.mean(forecast - test)

                # POCID
                diff_original = test[1:] - test[:-1]
                diff_pred = forecast[1:] - forecast[:-1]
                is_positive = (diff_original * diff_pred) > 0
                pocid = is_positive.sum() / len(is_positive) if len(is_positive) > 0 else 0.0

                # Score
                score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)

                # ── Save baseline result to CSV immediately ───────────────────
                csv_results_path = results_csv
                _bl_csv_exists = os.path.exists(csv_results_path)
                with open(csv_results_path, 'a', newline='') as _bl_csv:
                    _bl_w = csv.writer(_bl_csv)
                    if not _bl_csv_exists:
                        _bl_w.writerow([
                            "item_id", "store_id", "seed", "metric",
                            "window_size", "step_size", "threshold", "percentile",
                            "enable_edges", "enable_second_degree", "ablate_z",
                            "node_feature_mode",
                            "rmse", "mae", "bias", "r2_score", "pocid",
                            "train_time_s", "inference_time_s",
                        ])
                    _bl_w.writerow([
                        product_id, store_id, seed, "N/A",
                        "N/A", "N/A", "", "",
                        "N/A", "N/A", "baseline",
                        "N/A",
                        rmse, mae, bias, score, pocid,
                        f"{train_time:.2f}" if train_time is not None else "",
                        f"{infer_time:.4f}",
                    ])
                done_set.add(_bl_done_key)
                print(f"Baseline result written to CSV (RMSE={rmse:.4f})")

                # Store baseline results for inclusion as a line in the combined plot
                _bl_forecast = forecast
                _bl_t_losses = t_losses
                _bl_v_losses = v_losses
                _bl_rmse, _bl_mae, _bl_bias, _bl_score, _bl_pocid = rmse, mae, bias, score, pocid

            for config in all_configs:
                metric      = config['metric']
                thresholds  = config.get('thresholds',  [None])
                percentiles = config.get('percentiles', [None])

                results_by_w_s = {}

                is_threshold_mode = thresholds is not None and thresholds != [None]
                params   = thresholds if is_threshold_mode else percentiles
                iterator = itertools.product(
                    ABLATE_Z_VALUES, params, WINDOW_SIZES, STEP_SIZES,
                    ENABLE_EDGES_OPTS, ENABLE_SECOND_DEGREE_OPTS, node_feature_modes,
                )

                metric_type = infer_metric_type(metric)

                # Accumulated per-step neighbour map: threshold -> {step_idx: [nbr_ids]}
                # Populated by the first non-ablation run; reused by all subsequent traces.
                all_step_neighbours: dict = {}

                for ablate_z, param_val, window_size, step_size, enable_edges, enable_second_degree, node_feature_mode in iterator:
                    # When ablating, z is zeroed so the threshold has no effect; the
                    # node-feature mode is also irrelevant (no graph is consumed).
                    if ablate_z and param_val != params[0]:
                        continue
                    if ablate_z and node_feature_mode != node_feature_modes[0]:
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
                          f"2nd_degree={enable_second_degree}, node_features={node_feature_mode}")
                    print(f"{'='*60}")

                    # ── 1 & 2. Graph pipeline (skipped for ablation) ─────────
                    if ablate_z:
                        # GCN output is zeroed; building hundreds of sliding-window
                        # graphs would be pure waste.  Dummy placeholder graphs.
                        fixed_threshold = None
                        results_by_w_s[key]['threshold'] = fixed_threshold
                        _dummy_in_channels = {'stats': 8, 'catch22': 22,
                                              'catch24': 24}.get(
                            node_feature_mode, window_size)
                        _dummy = Data(
                            x=torch.zeros(1, _dummy_in_channels, dtype=torch.float32),
                            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                            edge_attr=torch.zeros(1, 1, dtype=torch.float32),
                            num_nodes=1,
                        )
                        pyg_train         = [_dummy] * (val_start_idx  - train_start_idx)
                        pyg_val           = [_dummy] * (test_start_idx - val_start_idx)
                        pyg_seed_graphs   = [_dummy] * lookback_window
                        pyg_future_graphs = [_dummy] * forecast_horizon
                        _inf_nx_graphs    = None
                    else:
                        # ── 1. Build per-window NX graphs ────────────────────
                        # Always use df_wide_scaled (per-product z-score, fit on train only).
                        # Distance metrics need scaling; similarity metrics (Spearman/Pearson)
                        # are scale-invariant so topology is unchanged.  Node features are
                        # already normalised, so node_scalers is never needed.
                        current_df_wide = df_wide_scaled
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
                        
                        _inf_nx_graphs = nx_graphs[-forecast_horizon:]

                        # ── 2. NX -> per-window PyG, align to timeline (per-day) ──
                        pyg_windows = build_pyg_graphs_from_nx_windows(
                            nx_graphs, current_df_wide, product_id,
                            window_size=window_size, step_size=step_size,
                            node_feature_mode=node_feature_mode,
                        )

    

                        T_global = current_df_wide.shape[1]
                        pyg_aligned_global = _align_pyg_windows_to_timeline(
                            pyg_windows, window_size=window_size,
                            step_size=step_size, T=T_global,
                        )
                        product_offset = T_global - len(df_p)
                        pyg_train = pyg_aligned_global[product_offset + train_start_idx:
                                                       product_offset + val_start_idx]
                        pyg_val   = pyg_aligned_global[product_offset + val_start_idx - lookback_window:
                                                       product_offset + test_start_idx]

                        seed_start = product_offset + test_start_idx - lookback_window
                        seed_end   = product_offset + test_start_idx
                        pyg_seed_graphs = pyg_aligned_global[seed_start:seed_end]

                        fut_start = product_offset + test_start_idx
                        fut_end   = fut_start + forecast_horizon
                        pyg_future_graphs = pyg_aligned_global[fut_start:fut_end]

                    # ── 3. Datasets / loaders (PER-STEP) ─────────────────────
                    use_pin_memory = torch.cuda.is_available()
                    train_dataset = GCNTimeSeriesDataset(
                        target_data=train_scaled,
                        exog_data=exog_train_scaled if EXOG_COLS else None,
                        lookback_window=lookback_window,
                        pyg_graphs=pyg_train,
                        graph_window_size=window_size,
                    )
                    train_loader = DataLoader(
                        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=use_pin_memory, collate_fn=collate_pyg_ts,
                    )
                    val_dataset = GCNTimeSeriesDataset(
                        target_data=val_scaled_ctx,
                        exog_data=exog_val_scaled_ctx if EXOG_COLS else None,
                        lookback_window=lookback_window,
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

                    in_channels   = pyg_train[0].x.shape[1]
                    ts_input_size = 1 + (len(EXOG_COLS) if EXOG_COLS else 0)
                    if ablate_z:
                        # Pure TDMLP — no graph input, no wasted zero columns.
                        model = AblationMLPForecaster(
                            ts_input_size=ts_input_size,
                            hidden_sizes=ABLATION_HIDDEN_SIZES,
                            dropout=DROPOUT,
                        ).to(device)
                    else:
                        model = SimpleGCNMLPForecaster(
                            in_channels=in_channels,
                            gcn_hidden=GCN_HIDDEN_SIZE,
                            d_g=GRAPH_EMBEDDINGS_DIM,
                            ts_input_size=ts_input_size,
                            lookback_window=lookback_window,
                            hidden_sizes=HIDDEN_SIZES,
                            horizon=1,
                            dropout=DROPOUT,
                        ).to(device)
                    criterion  = nn.MSELoss()
                    criterion2 = nn.MSELoss()
                    optimizer  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
                    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=PATIENCE // 3,
                    )

                    # ── 5. Checkpoint paths ──────────────────────────────────
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
                    az_tag = "_ablation" if ablate_z else ""
                    param_label = (f"th_{current_threshold}" if is_threshold_mode
                                   else f"pct_{current_percentile}")
                    base_name = (
                        f"best_gcnmlp_perstep_{prefix_star}{product_id}_{metric}"
                        f"_w{window_size}_s{step_size}_{param_label}_nf{node_feature_mode}"
                        f"{res_tag}{az_tag}_seed_{seed}"
                    )
                    best_model_path  = os.path.join(best_models_dir, f"{base_name}.pth")
                    history_path     = os.path.join(best_models_dir, f"{base_name}_history.pkl")
                    combined_pkl_path = os.path.join(best_models_dir, f"{base_name}_combined.pkl")
                    print(f"Resolved checkpoint: {best_model_path}")

                    # ── 6. Train ─────────────────────────────────────────────
                    print("Training new per-step GCN+MLP model...")
                    model, train_losses, val_losses, best_epoch, train_time = train_model(
                        seed=seed, epochs=EPOCHS, model=model,
                        train_loader=train_loader, val_loader=val_loader,
                        criterion=criterion, criterion2=criterion2,
                        optimizer=optimizer, device=device,
                        best_model_path=best_model_path,
                        scheduler=scheduler, patience=PATIENCE,
                    )
                    # Load best-epoch weights for inference
                    if os.path.exists(best_model_path):
                        print(f"Loading best weights from {best_model_path} for inference...")
                        model.load_state_dict(torch.load(best_model_path, map_location=device))

                    # ── 7. Recursive inference ───────────────────────────────
                    inf_threshold = fixed_threshold
                    print("Running Inference...")

                    if EXOG_COLS:
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

                    lag_col_indices = {}
                    for i, name in enumerate(EXOG_COLS):
                        if name.startswith(LAG_PREFIX):
                            try:
                                k = int(name[len(LAG_PREFIX):])
                            except ValueError:
                                continue
                            lag_col_indices[i] = k
                    rolling_mean_excl_col_indices = {}
                    for i, name in enumerate(EXOG_COLS):
                        if name.startswith(ROLLING_MEAN_EXCL_PREFIX):
                            try:
                                W = int(name[len(ROLLING_MEAN_EXCL_PREFIX):])
                            except ValueError:
                                continue
                            rolling_mean_excl_col_indices[i] = W
                    target_history_unscaled = np.concatenate([train, val]).astype(np.float32)

                    # ── Build per-step graph-save callback ───────────────────
                    if _inf_nx_graphs is not None and SAVE_INFERENCE_GRAPHS_PLOTS:
                        _param_label_plot = (f"th_{current_threshold}" if is_threshold_mode
                                             else f"pct_{current_percentile}")
                        _graph_plot_dir = os.path.join(
                            SCRIPT_DIR, 'graph_infered_plots', str(product_id),
                            f'seed_{seed}', metric, _param_label_plot, f"nf_{node_feature_mode}",
                        )
                        os.makedirs(_graph_plot_dir, exist_ok=True)
                        print(f"\nWill save {len(_inf_nx_graphs)} inference graph plots during inference...")

                        def _make_step_cb(graphs, plot_dir, lbl, w, s, pid, met, nf_mode):
                            def _cb(step_idx):
                                if step_idx < len(graphs):
                                    _title = (
                                        f"Product {pid} | {met} | {lbl} | "
                                        f"w{w}_s{s} | nf:{nf_mode} | inference step {step_idx + 1}"
                                    )
                                    _sp = os.path.join(
                                        plot_dir,
                                        f"graph_{met}_{lbl}_w{w}_s{s}_nf-{nf_mode}_step{step_idx + 1:04d}.html",
                                    )
                                    plot_networkx_plotly(G=graphs[step_idx], title=_title,
                                                         save_path=_sp, target_node=pid)
                            return _cb

                        _step_callback = _make_step_cb(
                            _inf_nx_graphs, _graph_plot_dir, _param_label_plot,
                            window_size, step_size, product_id, metric, node_feature_mode,
                        )
                    else:
                        _step_callback = None

                    graph_log = [] if RECORD_INFERENCE_GRAPHS else None
                    _inf_start = time.time()

                    # Std-scaled seed window for target-node feature patching:
                    # window_size values immediately before the test period,
                    # taken from df_wide_scaled (no leakage).
                    # Only applicable for the full GCN run (not ablation).
                    if not ablate_z:
                        _std_seed_end   = product_offset + test_start_idx
                        _std_seed_start = _std_seed_end - window_size
                        _target_std_seed = (
                            df_wide_scaled.loc[product_id].values[_std_seed_start:_std_seed_end]
                            .astype(np.float32)
                        )
                    else:
                        _target_std_seed = None

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
                        exog_cols=EXOG_COLS if EXOG_COLS else None,
                        graph_log_out=graph_log,
                        step_callback=_step_callback,
                        target_node_std_scaler=product_scalers[product_id] if not ablate_z else None,
                        target_node_std_seed=_target_std_seed,
                        node_feature_mode=node_feature_mode,
                    )
                    inference_time = time.time() - _inf_start

                    # ── Capture per-step neighbours for hover tooltip ─────────
                    # Build {step_idx: [nbr_ids]} for every forecast step and
                    # store it keyed by threshold so _trace_customdata in plots.py
                    # can attach it to the matching forecast trace's customdata.
                    if _inf_nx_graphs is not None and not ablate_z:
                        _step_nbrs: dict = {}
                        for _si, _G in enumerate(_inf_nx_graphs):
                            _step_nbrs[_si] = [n for n in _G.nodes() if n != product_id]
                        all_step_neighbours[fixed_threshold] = _step_nbrs

                    if RECORD_INFERENCE_GRAPHS and graph_log:
                        _igcsv = os.path.join(SCRIPT_DIR, "inference_graph_log.csv")
                        _igexists = os.path.exists(_igcsv)
                        with open(_igcsv, 'a', newline='') as _gf:
                            _gw = csv.writer(_gf)
                            if not _igexists:
                                _gw.writerow([
                                    "product_id", "store_id", "seed", "metric",
                                    "window_size", "step_size",
                                    "threshold", "percentile", "ablate_z",
                                    "step", "n_nodes", "src_node", "tgt_node", "edge_weight",
                                ])
                            for _row in graph_log:
                                _gw.writerow([
                                    product_id, store_id, seed, metric,
                                    window_size, step_size,
                                    current_threshold if current_threshold is not None else "",
                                    current_percentile if current_percentile is not None else "",
                                    ablate_z,
                                    _row['step'], _row['n_nodes'],
                                    _row['src_node'], _row['tgt_node'], _row['edge_weight'],
                                ])

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
                                  f"|e:{enable_edges}|2nd:{enable_second_degree}"
                                  f"|nf:{node_feature_mode}|az:{az_str}")

                    results_by_w_s[key]['forecasts'][label_name]    = forecast
                    results_by_w_s[key]['train_losses'][label_name] = train_losses
                    results_by_w_s[key]['val_losses'][label_name]   = val_losses
                    results_by_w_s[key]['rmse'][label_name]         = rmse
                    results_by_w_s[key]['mae'][label_name]          = mae
                    results_by_w_s[key]['bias'][label_name]         = bias
                    results_by_w_s[key]['score'][label_name]        = score
                    results_by_w_s[key]['pocid'][label_name]        = pocid

                    print(f"Finished {metric} @ {param_val} -> RMSE: {rmse}\n")

                    # ── Append to per-worker CSV ──────────────────────────────
                    csv_results_path = results_csv
                    file_exists = os.path.exists(csv_results_path)
                    with open(csv_results_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow([
                                "item_id", "store_id", "seed", "metric",
                                "window_size", "step_size", "threshold", "percentile",
                                "enable_edges", "enable_second_degree", "ablate_z",
                                "node_feature_mode",
                                "rmse", "mae", "bias", "r2_score", "pocid",
                                "train_time_s", "inference_time_s",
                            ])
                        writer.writerow([
                            product_id, store_id, seed, metric,
                            window_size, step_size,
                            current_threshold if current_threshold is not None else "",
                            current_percentile if current_percentile is not None else "",
                            enable_edges, enable_second_degree, ablate_z,
                            node_feature_mode,
                            rmse, mae, bias, score, pocid,
                            f"{train_time:.2f}" if train_time is not None else "",
                            f"{inference_time:.4f}",
                        ])

                # ── Per-metric combined plot ─────────────────────────────────
                train_index = df_p[DATE_COL][train_slice].values
                val_index   = df_p[DATE_COL][val_slice].values
                test_index  = df_p[DATE_COL][test_slice].values

                grouped_results = {}
                for (az, p, w, s), res_dicts in results_by_w_s.items():
                    key = (w, s)
                    if key not in grouped_results:
                        grouped_results[key] = {
                            'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                            'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {},
                        }
                    for k in grouped_results[key]:
                        grouped_results[key][k].update(res_dicts[k])

                for (w, s), res_dicts in grouped_results.items():
                    raw_str = ("_".join(map(str, thresholds))
                               if thresholds is not None and len(thresholds) > 0 and percentiles is None
                               else "_".join(map(str, percentiles)))
                    values_str = hashlib.md5(raw_str.encode()).hexdigest()[:8]

                    sub_dir = os.path.join(
                        grid_search_plots_dir, metric_type, f'window_{w}', f'step_{s}',
                        f'item_{product_id}', values_str,
                    )
                    os.makedirs(sub_dir, exist_ok=True)
                    save_plot_path = os.path.join(
                        sub_dir,
                        f"item_{product_id}_{metric}_seed_{seed}_all_configs.html",
                    )
                    emb_title = (
                        f'GCN+MLP (per-step) Forecasts + Ablation ({metric} | Seed={seed} | W={w} | S={s})'
                    )

                    if SAVE_PLOTS:
                        print(f"Saving combined plot to: {os.path.abspath(save_plot_path)}")
                        # Inject baseline as a reference line in the combined plot
                        res_dicts['forecasts']['baseline']    = _bl_forecast
                        res_dicts['train_losses']['baseline'] = _bl_t_losses
                        res_dicts['val_losses']['baseline']   = _bl_v_losses
                        res_dicts['rmse']['baseline']         = _bl_rmse
                        res_dicts['mae']['baseline']          = _bl_mae
                        res_dicts['bias']['baseline']         = _bl_bias
                        res_dicts['score']['baseline']        = _bl_score
                        res_dicts['pocid']['baseline']        = _bl_pocid
                        plot_results(
                            train, val, test, res_dicts['forecasts'],
                            train_index, val_index, test_index,
                            res_dicts['train_losses'], res_dicts['val_losses'],
                            metric=metric, embedding_strategy='gcn_perstep_mlp',
                            window_size=w, step_size=s, threshold=None, percentile=None,
                            target_col=TARGET_COL,
                            title=f'{emb_title} (Item={product_id})',
                            seed=seed, save_path=save_plot_path,
                            rmse=res_dicts['rmse'], mae=res_dicts['mae'],
                            bias=res_dicts['bias'], score=res_dicts['score'],
                            pocid=res_dicts['pocid'],
                            all_step_neighbours=all_step_neighbours if all_step_neighbours else None,
                        )
    
    experience_end_time = time.time()
    elapsed_time = experience_end_time - experience_start_time
    print(f"\nTotal time of experience : {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
    timing_summary_path = os.path.join(SCRIPT_DIR, "timing_summary.csv")
    timing_exists = os.path.exists(timing_summary_path)
    with open(timing_summary_path, 'a', newline='') as _tf:
        _tw = csv.writer(_tf)
        if not timing_exists:
            _tw.writerow(["run_timestamp", "total_elapsed_s", "total_elapsed_min",
                          "products", "seeds", "metrics"])
        _tw.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            f"{elapsed_time:.2f}",
            f"{elapsed_time/60:.2f}",
            str(PRODUCTS_TO_TEST),
            str(SEEDS),
        ])
    
    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["TOTAL_EXPERIMENT_TIME_SECONDS", f"{elapsed_time:.2f}"] + [""] * 17)
        
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
                    if 'threshold' not in res_df.columns or 'rmse' not in res_df.columns:
                        continue
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
