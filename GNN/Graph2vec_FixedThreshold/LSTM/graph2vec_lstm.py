import csv as _csv_mod
import csv
import glob as _glob_mod
import hashlib
import itertools
import os
import pickle
import random
import re
import subprocess
import sys
import threading
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths & Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

from lstm import LSTM
from graph2vecdataset import TimeSeriesDataset
from model_utils.utils import generate_exogenous_features, compute_metrics
from plots import plot_results
from generate_graph2vecwithadaptativethreshold import load_or_generate_embeddings, infer_metric_type
from train import train_model
from graph2vecinference_adaptativethreshold import graph2vec_inference

# Constants
FULL_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/data_andre_fulfilled.feather'))
TOP_DATA_PATH  = os.path.normpath(os.path.join(SCRIPT_DIR, '../../../dataset/top_12500.feather'))
DATE_COL   = 'date'
TARGET_COL = 'value'

SEEDS = [42, 1000, 26008, 555555, 213626, 907969, 5219788, 13451285, 23616558, 618626816]

PRODUCTS_TO_TEST = None

EXOG_COLS = [
    "day_of_week", "day_of_month", "week_of_year", "week_of_month",
    "month", "quarter", "is_weekend",
    #"lag_1", "lag_7", "lag_30",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_monday", "is_friday",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_pre_holiday_1", "is_pre_holiday_2", "is_pre_holiday_3", "is_pre_holiday_7",
    "is_post_holiday_1", "is_post_holiday_2", "is_post_holiday_3", "is_post_holiday_7",
    "is_bridge_day",
]

grid_configs = [
    {'metric': 'spearman', 'thresholds': [0.6, 0.65, 0.70]}
]

window_sizes              = [30]
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
SAVE_MODELS               = False
SAVE_PLOTS                = True
USE_EMBEDDINGS            = True
SAVE_EMBEDDINGS           = False
SAVE_INFERENCE_GRAPHS_PLOTS = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Parallelism config ────────────────────────────────────────────────────────
# PARALLEL_MODE: "product" → one subprocess per product-shard (recommended)
#                "seed"    → one subprocess per seed (each handles all products)
# MAX_CONCURRENT caps simultaneous processes.
# N_PRODUCT_WORKERS: how many product shards to create (product mode only).
# PRODUCTS_END: only distribute PRODUCTS_TO_TEST[:PRODUCTS_END]; None = all.
# NO_MERGE=True skips merging the per-worker CSVs into the canonical {metric}.csv.
PARALLEL           = Truefr
PARALLEL_MODE      = "product"  # "seed" or "product"
MAX_CONCURRENT     = 6
N_PRODUCT_WORKERS  = 6
PRODUCTS_END       = 60
NO_MERGE           = False


# ── CSV merge helper ──────────────────────────────────────────────────────────
def _merge_worker_csvs(metrics):
    """Merge per-worker {metric}_w*.csv files into the canonical {metric}.csv."""
    for metric in metrics:
        partial_files = sorted(_glob_mod.glob(os.path.join(SCRIPT_DIR, f"{metric}_w*.csv")))
        if not partial_files:
            continue
        seen   = set()
        rows   = []
        header = None
        for p in partial_files:
            try:
                with open(p, newline="") as f:
                    reader = csv.DictReader(f)
                    header = reader.fieldnames or header
                    for row in reader:
                        key = (
                            row.get("product_id", ""),       row.get("store_id", ""),
                            row.get("seed", ""),             row.get("metric", ""),
                            row.get("window_size", ""),      row.get("step_size", ""),
                            row.get("threshold", ""),        row.get("percentile", ""),
                            row.get("enable_edges", ""),     row.get("enable_second_degree", ""),
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
    to max_concurrent at a time, then optionally merge per-worker CSVs.
    """
    semaphore = threading.Semaphore(max_concurrent)
    results   = {}
    threads   = []

    def _run(wid, nw, extra_env, log_name):
        with semaphore:
            env = {
                **os.environ,
                "G2V_WORKER_ID":        str(wid),
                "G2V_NUM_WORKERS":      str(nw),
                "OMP_NUM_THREADS":      "1",
                "MKL_NUM_THREADS":      "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS":  "1",
                **extra_env,
            }
            cmd      = [sys.executable, __file__]
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
            status  = "OK" if rc == 0 else f"FAILED(rc={rc})"
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
        (wid, 1, {"G2V_SEED": str(seed)}, f"log_seed{seed}.txt")
        for wid, seed in enumerate(seeds)
    ]
    _spawn_workers(jobs, max_concurrent, metrics, no_merge=no_merge)


# ── Main runner ───────────────────────────────────────────────────────────────
def main():
    worker_id   = int(os.environ.get("G2V_WORKER_ID", "0"))
    num_workers = int(os.environ.get("G2V_NUM_WORKERS", "1"))

    if num_workers > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # ── Parallel launcher (coordinator only) ─────────────────────────────────
    is_worker = num_workers > 1 or "G2V_SEED" in os.environ
    if PARALLEL and not is_worker:
        metrics = [c['metric'] for c in grid_configs]
        if USE_EMBEDDINGS:
            metrics = ['no_emb'] + metrics
        if PARALLEL_MODE == "seed":
            _run_parallel_seeds(
                seeds=SEEDS, max_concurrent=MAX_CONCURRENT,
                metrics=metrics, no_merge=NO_MERGE,
            )
        else:
            _run_parallel_products(
                num_workers=N_PRODUCT_WORKERS, max_concurrent=MAX_CONCURRENT,
                metrics=metrics, no_merge=NO_MERGE,
            )
        return

    seeds_to_run = ([int(os.environ["G2V_SEED"])] if "G2V_SEED" in os.environ
                    else list(SEEDS))

    # ── Load and preprocess data ──────────────────────────────────────────────
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

    L = len(df_wide_global.columns)
    forecast_horizon_global = 152
    val_size_global         = 154
    train_size_global       = 455
    global_train_start_idx  = max(0, L - forecast_horizon_global - val_size_global - train_size_global)
    global_val_start_idx    = L - forecast_horizon_global - val_size_global

    product_scalers = {}
    train_df_wide   = df_wide_global.iloc[:, global_train_start_idx:global_val_start_idx]
    df_wide_scaled  = df_wide_global.copy()
    for item_id_iter in df_wide_global.index:
        z_scaler = StandardScaler()
        z_scaler.fit(train_df_wide.loc[item_id_iter].values.reshape(-1, 1))
        product_scalers[item_id_iter] = z_scaler
        df_wide_scaled.loc[item_id_iter] = z_scaler.transform(
            df_wide_global.loc[item_id_iter].values.reshape(-1, 1)
        ).flatten()

    # ── Build product list ────────────────────────────────────────────────────
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

    if PRODUCTS_END is not None:
        PRODUCTS_TO_TEST = PRODUCTS_TO_TEST[:PRODUCTS_END]

    # ── Product sharding ──────────────────────────────────────────────────────
    n_products_total = len(PRODUCTS_TO_TEST)
    PRODUCTS_TO_TEST = PRODUCTS_TO_TEST[worker_id::num_workers]
    print(f"Worker {worker_id}/{num_workers}: processing "
          f"{len(PRODUCTS_TO_TEST)} / {n_products_total} products  |  seeds={seeds_to_run}")

    worker_csv_suffix = f"_w{worker_id}" if num_workers > 1 else ""

    timings_csv_path = os.path.join(SCRIPT_DIR, f"timings{worker_csv_suffix}.csv")
    product_times    = []
    total_t0         = time.time()

    for product_id, store_id in PRODUCTS_TO_TEST:
        print(f"\n{'='*80}")
        print(f"PROCESSING PRODUCT {product_id} FOR STORE {store_id}")
        print(f"{'='*80}\n")

        df_p = (
            full_df[(full_df['item_id'] == product_id) & (full_df['store_id'] == store_id)]
            .sort_values(DATE_COL).reset_index(drop=True)
        )

        forecast_horizon = 152
        seq_length       = 30
        train_size       = 455
        val_size         = 154
        lookback_window  = 7
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
            exog_scaler       = MinMaxScaler()
            exog_train_scaled = exog_scaler.fit_transform(df_p[EXOG_COLS][train_slice].values)
            exog_val_scaled   = exog_scaler.transform(df_p[EXOG_COLS][val_slice].values)
            exog_test_scaled  = exog_scaler.transform(df_p[EXOG_COLS][test_slice].values)
        else:
            exog_train_scaled = exog_val_scaled = exog_test_scaled = None
            exog_scaler = None

        product_t0 = time.time()

        for seed in seeds_to_run:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            print(f"\n--- RUNNING WITH SEED {seed} ---\n")

            grid_search_plots_dir = os.path.join(
                SCRIPT_DIR, 'grid_search_plots',
                f'seed_{seed}', f'product_{product_id}_store_{store_id}',
            )
            best_models_seed_dir = os.path.join(SCRIPT_DIR, 'best_models', f'seed_{seed}')
            os.makedirs(grid_search_plots_dir, exist_ok=True)
            os.makedirs(best_models_seed_dir, exist_ok=True)

            all_configs = [{'metric': 'no_emb'}] + grid_configs if USE_EMBEDDINGS else [{'metric': 'no_emb'}]
            base_forecast, base_train_losses, base_val_losses = None, None, None
            base_rmse, base_mae, base_bias, base_score, base_pocid = None, None, None, None, None

            for config in all_configs:
                metric      = config['metric']
                thresholds  = config.get('thresholds', [None])
                percentiles = config.get('percentiles', [None])

                results_by_w_s = {}

                if metric == 'no_emb':
                    is_threshold_mode = False
                    iterator = [(None, 15, 1, False, False)]
                else:
                    is_threshold_mode = thresholds is not None and thresholds != [None]
                    params   = thresholds if is_threshold_mode else percentiles
                    iterator = itertools.product(
                        params, window_sizes, step_sizes,
                        enable_edges_opts, enable_second_degree_opts,
                    )

                for param_val, window_size, step_size, enable_edges, enable_second_degree in iterator:
                    use_embeddings = (metric != 'no_emb')

                    current_threshold  = param_val if use_embeddings and is_threshold_mode else None
                    current_percentile = param_val if use_embeddings and not is_threshold_mode else None

                    key = (param_val, window_size, step_size)
                    if key not in results_by_w_s:
                        results_by_w_s[key] = {
                            'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                            'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {},
                            'threshold': None,
                        }
                        if metric != 'no_emb' and base_forecast is not None:
                            results_by_w_s[key]['forecasts']["No Embeddings"]    = base_forecast
                            results_by_w_s[key]['train_losses']["No Embeddings"] = base_train_losses
                            results_by_w_s[key]['val_losses']["No Embeddings"]   = base_val_losses
                            results_by_w_s[key]['rmse']["No Embeddings"]         = base_rmse
                            results_by_w_s[key]['mae']["No Embeddings"]          = base_mae
                            results_by_w_s[key]['bias']["No Embeddings"]         = base_bias
                            results_by_w_s[key]['score']["No Embeddings"]        = base_score
                            results_by_w_s[key]['pocid']["No Embeddings"]        = base_pocid

                    print(f"\n{'='*60}")
                    if use_embeddings:
                        param_str = (f"threshold={current_threshold}" if is_threshold_mode
                                     else f"percentile={current_percentile}")
                        print(f"Running Experiment: metric={metric}, {param_str}, "
                              f"window_size={window_size}, enable_edges={enable_edges}, "
                              f"2nd_degree={enable_second_degree}")
                    else:
                        print("Running Experiment: BASELINE (no graph embeddings)")
                    print(f"{'='*60}")

                    fixed_threshold    = None
                    aligned_embeddings = None
                    graph2vec_model    = None
                    embedding_dim      = 0
                    emb_train = emb_val = None
                    current_df_wide    = df_wide_global

                    if use_embeddings:
                        metric_type = infer_metric_type(metric)

                        distance_metrics = [
                            'euclidean', 'manhattan', 'hamming', 'amplitude_offset',
                            'slope_consistency', 'phase_invariance', 'dtw', 'cid',
                            'lorentzian', 'sbd', 'msm', 'edr', 'lcss',
                        ]
                        current_df_wide = df_wide_scaled if metric in distance_metrics else df_wide_global

                        (graph_embeddings, graph2vec_model, csv_path,
                         _, _, fixed_threshold) = load_or_generate_embeddings(
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
                            save_embeddings=SAVE_EMBEDDINGS,
                        )
                        print(f"Resolved graph threshold={current_threshold}: {fixed_threshold}")
                        print(f"Embedding file: {csv_path}")
                        results_by_w_s[key]['threshold'] = fixed_threshold

                        embedding_dim      = graph_embeddings.shape[1] if len(graph_embeddings.shape) > 1 else 1
                        padding            = np.zeros((window_size - 1, embedding_dim))
                        aligned_embeddings = np.vstack([padding, graph_embeddings])
                        emb_train          = aligned_embeddings[train_slice]
                        emb_val            = aligned_embeddings[val_slice]

                    input_size = 1 + (len(EXOG_COLS) if EXOG_COLS else 0) + embedding_dim

                    use_pin_memory = torch.cuda.is_available()
                    train_dataset  = TimeSeriesDataset(
                        target_data=train_scaled,
                        exog_data=exog_train_scaled if EXOG_COLS else None,
                        seq_length=seq_length,
                        embeddings=emb_train,
                        graph_window_size=window_size,
                    )
                    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                             shuffle=False, pin_memory=use_pin_memory)

                    val_dataset = TimeSeriesDataset(
                        target_data=val_scaled,
                        exog_data=exog_val_scaled if EXOG_COLS else None,
                        seq_length=seq_length,
                        embeddings=emb_val,
                        graph_window_size=window_size,
                    )
                    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                           shuffle=False, pin_memory=use_pin_memory)

                    torch.manual_seed(seed)
                    np.random.seed(seed % (2**32))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)

                    model     = LSTM(input_size=input_size, hidden_size=HIDDEN_SIZE,
                                    num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
                    criterion  = nn.MSELoss()
                    criterion2 = nn.MSELoss()
                    optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=PATIENCE // 3,
                    )

                    model_dir_label = ("no_emb" if not use_embeddings
                                      else (f"th{current_threshold}" if is_threshold_mode
                                            else f"pct{current_percentile}"))
                    best_models_dir = os.path.join(
                        best_models_seed_dir, str(window_size), str(step_size),
                        metric, model_dir_label,
                    )
                    os.makedirs(best_models_dir, exist_ok=True)

                    if use_embeddings:
                        if csv_path:
                            csv_basename    = os.path.basename(csv_path)
                            base_name       = csv_basename.replace('embeddings_', f'best_lstm_{product_id}_')
                            base_name       = (base_name.replace('.csv', f'_res_{MODEL_TYPE}.pth')
                                               if USE_RESIDUALS else base_name.replace('.csv', '.pth'))
                            hist_name       = base_name.replace('.pth', '_history.pkl')
                            best_model_path = os.path.join(best_models_dir, base_name)
                            history_path    = os.path.join(best_models_dir, hist_name)
                        else:
                            prefix_star = "" if enable_edges else "star_"
                            if enable_second_degree:
                                prefix_star = "2nddegree_" + prefix_star
                            prefix = (f"best_lstm_{prefix_star}{product_id}_{metric}_res_{MODEL_TYPE}"
                                      if USE_RESIDUALS else f"best_lstm_{prefix_star}{product_id}_{metric}")
                            param_label     = (f"th_{current_threshold}" if is_threshold_mode
                                               else f"pct_{current_percentile}")
                            best_model_path = os.path.join(
                                best_models_dir,
                                f'{prefix}_{window_size}_{step_size}_{param_label}_seed_{seed}.pth',
                            )
                            history_path = os.path.join(
                                best_models_dir,
                                f'{prefix}_{window_size}_{step_size}_{param_label}_seed_{seed}_history.pkl',
                            )
                    else:
                        best_model_path = os.path.join(
                            best_models_dir, f'best_lstm_{product_id}_no_emb_seed_{seed}.pth',
                        )
                        history_path = os.path.join(
                            best_models_dir, f'best_lstm_{product_id}_no_emb_seed_{seed}_history.pkl',
                        )

                    print(f"Resolved LSTM checkpoint: {best_model_path}")

                    if os.path.exists(best_model_path) and os.path.exists(history_path):
                        print(f"Loading existing model from {best_model_path}...")
                        model.load_state_dict(torch.load(best_model_path, map_location=device))
                        with open(history_path, 'rb') as f:
                            history      = pickle.load(f)
                            train_losses = history['train_losses']
                            val_losses   = history['val_losses']
                    else:
                        print("Training new model...")
                        model, train_losses, val_losses, best_epoch, train_time = train_model(
                            seed=seed, epochs=EPOCHS, model=model,
                            train_loader=train_loader, val_loader=val_loader,
                            exog_cols=EXOG_COLS, criterion=criterion, criterion2=criterion2,
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

                    exog_test_data = df_p[EXOG_COLS][test_slice].values if EXOG_COLS else None
                    inf_threshold  = fixed_threshold if use_embeddings and fixed_threshold is not None else None

                    print("Running Inference...")
                    result_tuple = graph2vec_inference(
                        metric=metric, window_size=window_size, step_size=step_size,
                        model=model,
                        df=df_p, df_wide=current_df_wide, cat_labels=cat_labels_dict,
                        date_col=DATE_COL,
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
                        percentile=current_percentile if not is_threshold_mode else None,
                        threshold=inf_threshold,
                        create_plots=False,
                        return_graphs=SAVE_INFERENCE_GRAPHS_PLOTS,
                    )

                    if SAVE_INFERENCE_GRAPHS_PLOTS:
                        forecast, inference_time, inference_nx_graphs = result_tuple
                        graph_plot_dir = os.path.join(
                            SCRIPT_DIR, 'graph_infered_plots', str(product_id), f'seed_{seed}',
                        )
                        os.makedirs(graph_plot_dir, exist_ok=True)

                        utils_path = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'GraphAnalysis'))
                        if utils_path not in sys.path:
                            sys.path.append(utils_path)
                        try:
                            from utils import plot_dynamic_graphs
                        except ImportError:
                            from GNN.Graph2vec_FixedThreshold.LSTM.utils import plot_dynamic_graphs

                        print(f"\nSaving inference graph plots for product {product_id} (Seed: {seed}) ...")
                        plot_dynamic_graphs(
                            graphs=inference_nx_graphs,
                            product_id=product_id,
                            metric=metric,
                            plot_dir=graph_plot_dir,
                            residuals=USE_RESIDUALS,
                            enable_edges_within_star=enable_edges,
                            enable_second_degree=enable_second_degree,
                            num_plots=None,
                            window_size=window_size,
                            step_size=step_size,
                            threshold=inf_threshold,
                            percentile=current_percentile,
                        )
                    else:
                        forecast, inference_time = result_tuple

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

                    if use_embeddings:
                        param_str_label = (f"th:{current_threshold}" if is_threshold_mode
                                           else f"pct:{current_percentile} (val:{th_str})")
                        label_name = (f"{param_str_label}|w:{window_size}|st:{step_size}"
                                      f"|e:{enable_edges}|2nd:{enable_second_degree}")
                    else:
                        label_name = "No Embeddings"

                    if metric == 'no_emb':
                        base_forecast, base_train_losses, base_val_losses = forecast, train_losses, val_losses
                        base_rmse, base_mae, base_bias = rmse, mae, bias
                        base_score, base_pocid         = score, pocid

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
                    csv_results_path = os.path.join(SCRIPT_DIR, f"{metric}{worker_csv_suffix}.csv")
                    file_exists      = os.path.exists(csv_results_path)
                    with open(csv_results_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow([
                                "product_id", "store_id", "seed", "metric",
                                "window_size", "step_size", "threshold", "percentile",
                                "enable_edges", "enable_second_degree",
                                "rmse", "mae", "bias", "r2_score", "pocid",
                            ])
                        writer.writerow([
                            product_id, store_id, seed, metric,
                            15 if metric == 'no_emb' else window_size,
                            1  if metric == 'no_emb' else step_size,
                            current_threshold  if use_embeddings else "",
                            current_percentile if use_embeddings else "",
                            enable_edges        if use_embeddings else "",
                            enable_second_degree if use_embeddings else "",
                            rmse, mae, bias, score, pocid,
                        ])

                # ── Per-metric plots ──────────────────────────────────────────
                train_index = df_p[DATE_COL][train_slice].values
                val_index   = df_p[DATE_COL][val_slice].values
                test_index  = df_p[DATE_COL][test_slice].values

                if metric == 'no_emb':
                    values_str    = "no_thresholds"
                    sub_dir       = os.path.join(grid_search_plots_dir, 'no_emb',
                                                 f'window_{15}', f'step_{1}',
                                                 f'item_{product_id}', values_str)
                    os.makedirs(sub_dir, exist_ok=True)
                    save_plot_path = os.path.join(
                        sub_dir, f"item_{product_id}_store_{store_id}_no_emb_seed_{seed}.html",
                    )
                    emb_title = f'Baseline LSTM Forecast (No Embeddings | Seed={seed})'
                    if SAVE_PLOTS:
                        print(f"Saving combined plot to: {os.path.abspath(save_plot_path)}")
                        plot_results(
                            train, val, test,
                            results_by_w_s[(None, 15, 1)]['forecasts'],
                            train_index, val_index, test_index,
                            results_by_w_s[(None, 15, 1)]['train_losses'],
                            results_by_w_s[(None, 15, 1)]['val_losses'],
                            metric=metric, embedding_strategy='graph2vec',
                            window_size=15, step_size=1, threshold=None, percentile=None,
                            target_col=TARGET_COL,
                            title=f'{emb_title} (Item={product_id})', seed=seed,
                            save_path=save_plot_path,
                            rmse=results_by_w_s[(None, 15, 1)]['rmse'],
                            mae=results_by_w_s[(None, 15, 1)]['mae'],
                            bias=results_by_w_s[(None, 15, 1)]['bias'],
                            score=results_by_w_s[(None, 15, 1)]['score'],
                            pocid=results_by_w_s[(None, 15, 1)]['pocid'],
                        )
                else:
                    metric_type     = infer_metric_type(metric)
                    grouped_results = {}
                    for (p, w, s), res_dicts in results_by_w_s.items():
                        gkey = (w, s)
                        if gkey not in grouped_results:
                            grouped_results[gkey] = {
                                'forecasts': {}, 'train_losses': {}, 'val_losses': {},
                                'rmse': {}, 'mae': {}, 'bias': {}, 'score': {}, 'pocid': {},
                            }
                        for k in grouped_results[gkey]:
                            grouped_results[gkey][k].update(res_dicts[k])

                    for (w, s), res_dicts in grouped_results.items():
                        if thresholds is not None and len(thresholds) > 0 and percentiles is None:
                            raw_str = "_".join(map(str, thresholds))
                        else:
                            raw_str = "_".join(map(str, percentiles))
                        values_str = hashlib.md5(raw_str.encode()).hexdigest()[:8]
                        sub_dir    = os.path.join(
                            grid_search_plots_dir, metric_type,
                            f'window_{w}', f'step_{s}', f'item_{product_id}', values_str,
                        )
                        os.makedirs(sub_dir, exist_ok=True)
                        save_plot_path = os.path.join(
                            sub_dir, f"item_{product_id}_{metric}_seed_{seed}_all_configs.html",
                        )
                        emb_title = f'Graph2Vec Forecasts ({metric} | Seed={seed} | W={w} | S={s})'
                        if SAVE_PLOTS:
                            print(f"Saving combined plot to: {os.path.abspath(save_plot_path)}")
                            plot_results(
                                train, val, test, res_dicts['forecasts'],
                                train_index, val_index, test_index,
                                res_dicts['train_losses'], res_dicts['val_losses'],
                                metric=metric, embedding_strategy='graph2vec',
                                window_size=w, step_size=s, threshold=None, percentile=None,
                                target_col=TARGET_COL,
                                title=f'{emb_title} (Item={product_id})', seed=seed,
                                save_path=save_plot_path,
                                rmse=res_dicts['rmse'],   mae=res_dicts['mae'],
                                bias=res_dicts['bias'],   score=res_dicts['score'],
                                pocid=res_dicts['pocid'],
                            )

        product_elapsed = time.time() - product_t0
        product_times.append((product_id, store_id, product_elapsed))
        print(f"\n[TIMING] Product {product_id} | store {store_id}: "
              f"{product_elapsed:.1f} s ({product_elapsed/60:.2f} min)")

    # ── Total timing & persist ────────────────────────────────────────────────
    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    print(f"  TOTAL: {total_elapsed:.1f} s  ({total_elapsed/60:.2f} min)  "
          f"across {len(PRODUCTS_TO_TEST)} products")

    with open(timings_csv_path, "w", newline="") as fh:
        w = _csv_mod.writer(fh)
        w.writerow(["product_id", "store_id", "seconds"])
        for pid, sid, sec in product_times:
            w.writerow([pid, sid, f"{sec:.3f}"])
        w.writerow(["TOTAL_ALL", "", f"{total_elapsed:.3f}"])
    print(f"  Timings written to: {timings_csv_path}")

    # ── Correlation plots (one per metric CSV) ────────────────────────────────
    if SAVE_PLOTS:
        import matplotlib.pyplot as plt
        print("\nGenerating Correlation Plots across all collected CSVs...")
        for csv_file in os.listdir(SCRIPT_DIR):
            if not csv_file.endswith('.csv') or csv_file == 'no_emb.csv':
                continue
            metric_name = csv_file.replace('.csv', '')
            csv_path    = os.path.join(SCRIPT_DIR, csv_file)
            try:
                res_df = pd.read_csv(csv_path)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(f'Threshold vs RMSE and MAE | Metric: {metric_name}', fontsize=16)

                x_col     = 'threshold' if res_df['threshold'].notna().any() else 'percentile'
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

                plot_save_path = os.path.join(SCRIPT_DIR, f"{metric_name}_correlation_plot.png")
                plt.tight_layout()
                plt.savefig(plot_save_path)
                plt.close()
                print(f"Saved correlation plot for {metric_name} at {plot_save_path}")
            except Exception as e:
                print(f"Failed to generate plot for {csv_file}: {e}")


if __name__ == '__main__':
    main()
