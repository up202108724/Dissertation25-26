import csv as _csv_mod
import csv
import glob as _glob_mod
import itertools
import os
import pickle
import random
import subprocess
import sys
import threading
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths & Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from Baselines.LSTM.lstm import LSTM
from model_utils.utils import generate_exogenous_features, compute_metrics
from Graph2vec.LSTM.graph2vecdataset import TimeSeriesDataset
from Graph2vec.LSTM.train import train_model
from Graph2vec.LSTM.graph2vecinference_adaptativethreshold import graph2vec_inference
from Graph2vec.generate_graph2vecwithadaptativethreshold import load_or_generate_embeddings
from Graph2vec.graph_utils import infer_metric_type
from Graph2vec.experiment_utils import (
    load_panel, build_product_scalers, select_df_wide, split_and_scale,
    new_result_bucket, record_result, merge_result_buckets, config_label, param_grid_hash,
    plot_threshold_correlations,
)
from model_utils.plots import plot_results

# Constants
FULL_DATA_PATH = os.path.join(REPO_ROOT, 'dataset', 'data_andre_fulfilled.feather')
TOP_DATA_PATH  = os.path.join(REPO_ROOT, 'dataset', 'top_12500.feather')
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
PARALLEL           = True
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
    full_df, cat_labels_dict, df_wide_global = load_panel(
        FULL_DATA_PATH, EXOG_COLS, generate_exogenous_features, DATE_COL, TARGET_COL,
    )
    product_scalers, df_wide_scaled = build_product_scalers(
        df_wide_global, train_size=455, val_size=154, forecast_horizon=152,
    )

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

        sp = split_and_scale(df_p, TARGET_COL, EXOG_COLS, train_size, val_size, forecast_horizon)
        test_start_idx, val_start_idx = sp.test_start_idx, sp.val_start_idx
        train_slice, val_slice, test_slice = sp.train_slice, sp.val_slice, sp.test_slice
        train, val, test = sp.train, sp.val, sp.test
        scaler, train_scaled, val_scaled, test_scaled = sp.scaler, sp.train_scaled, sp.val_scaled, sp.test_scaled
        exog_scaler, exog_train_scaled = sp.exog_scaler, sp.exog_train_scaled
        exog_val_scaled, exog_test_scaled = sp.exog_val_scaled, sp.exog_test_scaled

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

            # No-graph baselines are produced by the Baselines/ folder; only graph configs run here.
            for config in grid_configs:
                metric      = config['metric']
                thresholds  = config.get('thresholds', [None])
                percentiles = config.get('percentiles', [None])

                results_by_w_s = {}

                is_threshold_mode = thresholds is not None and thresholds != [None]
                params   = thresholds if is_threshold_mode else percentiles
                iterator = itertools.product(
                    params, window_sizes, step_sizes,
                    enable_edges_opts, enable_second_degree_opts,
                )

                for param_val, window_size, step_size, enable_edges, enable_second_degree in iterator:
                    current_threshold  = param_val if is_threshold_mode else None
                    current_percentile = param_val if not is_threshold_mode else None

                    key = (param_val, window_size, step_size)
                    if key not in results_by_w_s:
                        results_by_w_s[key] = new_result_bucket()

                    print(f"\n{'='*60}")
                    param_str = (f"threshold={current_threshold}" if is_threshold_mode
                                 else f"percentile={current_percentile}")
                    print(f"Running Experiment: metric={metric}, {param_str}, "
                          f"window_size={window_size}, enable_edges={enable_edges}, "
                          f"2nd_degree={enable_second_degree}")
                    print(f"{'='*60}")

                    metric_type = infer_metric_type(metric)
                    current_df_wide = select_df_wide(metric, df_wide_global, df_wide_scaled)

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
                        output_dir=SCRIPT_DIR,
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

                    model_dir_label = (f"th{current_threshold}" if is_threshold_mode
                                       else f"pct{current_percentile}")
                    best_models_dir = os.path.join(
                        best_models_seed_dir, str(window_size), str(step_size),
                        metric, model_dir_label,
                    )
                    os.makedirs(best_models_dir, exist_ok=True)

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
                    inf_threshold  = fixed_threshold

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
                        node_embeddings=aligned_embeddings,
                        graph2vec_model=graph2vec_model,
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

                        from model_utils.plots import plot_dynamic_graphs

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

                    label_name = config_label(
                        is_threshold_mode, current_threshold, current_percentile, inf_threshold,
                        window_size, step_size, enable_edges, enable_second_degree,
                    )
                    record_result(results_by_w_s[key], label_name, dict(
                        forecasts=forecast, train_losses=train_losses, val_losses=val_losses,
                        rmse=rmse, mae=mae, bias=bias, score=score, pocid=pocid,
                    ))

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
                            window_size, step_size,
                            current_threshold, current_percentile,
                            enable_edges, enable_second_degree,
                            rmse, mae, bias, score, pocid,
                        ])

                # ── Per-metric plots ──────────────────────────────────────────
                train_index = df_p[DATE_COL][train_slice].values
                val_index   = df_p[DATE_COL][val_slice].values
                test_index  = df_p[DATE_COL][test_slice].values

                metric_type     = infer_metric_type(metric)
                grouped_results = {}
                for (p, w, s), res_dicts in results_by_w_s.items():
                    merge_result_buckets(grouped_results.setdefault((w, s), {}), res_dicts)

                for (w, s), res_dicts in grouped_results.items():
                    values_str = param_grid_hash(thresholds, percentiles)
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
        plot_threshold_correlations(SCRIPT_DIR)


if __name__ == '__main__':
    main()
