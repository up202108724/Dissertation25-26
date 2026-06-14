import copy
import sys
import os
import time
import csv
import glob as _glob_mod
import subprocess
import threading
import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.abspath('..'))
from plots import plot_results
from utils import generate_exogenous_features
from lstm import LSTM
from dataset import TimeSeriesDataset
from lstm_train import train_model, train_model_combined, train_model_expanding_window, train_model_sliding_window
#from train import train_model_selected_epochs
from lstm_inference import recursive_inference
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ── Constants (same defaults as LSTM_GCN/main.py) ──────────────────────────
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../dataset/data_andre_classified.feather'))
TOP_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../dataset/top_12500.feather'))
DATE_COL = 'date'
TARGET_COL = 'value'


val_size = 61
forecast_horizon = 153
train_size = 761 - val_size - forecast_horizon
lookback_window = 30

# Walk-forward CV folds for expanding/sliding window strategies.
# Pool = train+val = 761 - forecast_horizon = 608 days.
# initial_train_size = 365 -> first fold sees a full seasonal year.
# val_step_size = 81 -> the 243 remaining days tile into 3 equal folds whose
# last validation block ends exactly at the test boundary (no recent data wasted).
cv_initial_train_size = 365
cv_val_step_size = 81

BATCH_SIZE = 32

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

batch_size = 32
hidden_size = 32
num_layers = 1
dropout = 0.0
EPOCHS = 1000
LEARNING_RATE = 0.001
#seeds = [57]

#SEEDS = [42, 1000, 26008, 555555,213626, 907969, 5219788, 13451285, 23616558, 618626816]  # Add more seeds as needed
SEEDS = [42, 1000, 26008]  # Add more seeds as needed
loss_type = 'MSELoss'

RESULTS_CSV_NAME = 'lstm_strategies_results.csv'

# ── Parallelism config ──────────────────────────────────────────────────
# PARALLEL_MODE: "product" → one subprocess per product-shard (recommended)
#                "seed"    → one subprocess per seed (each handles all products)
# MAX_CONCURRENT caps simultaneous processes (Ryzen 7 5600X: 6 physical cores).
# N_PRODUCT_WORKERS: how many product shards to create (product mode only).
# PRODUCTS_END: only distribute products[:PRODUCTS_END]; None = all.
# NO_MERGE=True skips merging the per-worker CSVs into the canonical CSV.
PARALLEL = True
PARALLEL_MODE = "product"   # "seed" or "product"
MAX_CONCURRENT = 6
N_PRODUCT_WORKERS = 6       # product mode: products / N_PRODUCT_WORKERS each
PRODUCTS_END = None         # exclude products[PRODUCTS_END:] from the parallel run
NO_MERGE = False


# ─────────────────────────────────────────────────────────────────────────
# Parallel launchers (one subprocess per product-shard or per seed)
# ─────────────────────────────────────────────────────────────────────────
def _merge_worker_csvs():
    """Merge per-worker ``lstm_strategies_results_w*.csv`` files into the canonical CSV."""
    partial_files = sorted(_glob_mod.glob(
        os.path.join(SCRIPT_DIR, RESULTS_CSV_NAME.replace('.csv', '_w*.csv'))
    ))
    if not partial_files:
        return
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
                        row.get("item_id", ""), row.get("store_id", ""),
                        row.get("seed", ""), row.get("strategy", ""),
                    )
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
        except Exception as e:
            print(f"  Warning: could not read {p}: {e}")
    if header is None:
        return
    out_path = os.path.join(SCRIPT_DIR, RESULTS_CSV_NAME)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(rows)} rows from {len(partial_files)} files -> {out_path}")


def _spawn_workers(jobs, max_concurrent, no_merge=False):
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
                "LSTM_WORKER_ID": str(wid),
                "LSTM_NUM_WORKERS": str(nw),
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
        print(f"\nMerging partial CSVs -> {RESULTS_CSV_NAME}")
        _merge_worker_csvs()


def _run_parallel_products(num_workers, max_concurrent, no_merge=False):
    """One subprocess per disjoint product slice; each runs all SEEDS for its shard."""
    jobs = [
        (wid, num_workers, {}, f"log_worker{wid}_of{num_workers}.txt")
        for wid in range(num_workers)
    ]
    _spawn_workers(jobs, max_concurrent, no_merge=no_merge)


def _run_parallel_seeds(seeds, max_concurrent, no_merge=False):
    """One subprocess per seed; each handles all products for that single seed."""
    jobs = [
        (wid, 1, {"LSTM_SEED": str(seed)}, f"log_seed{seed}.txt")
        for wid, seed in enumerate(seeds)
    ]
    _spawn_workers(jobs, max_concurrent, no_merge=no_merge)


# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    worker_id   = int(os.environ.get("LSTM_WORKER_ID", "0"))
    num_workers = int(os.environ.get("LSTM_NUM_WORKERS", "1"))

    # Limit PyTorch's intra-op thread pool to 1 when running as a product-shard
    # worker so the processes share the cores instead of all claiming them.
    if num_workers > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # ── Parallel launcher (coordinator only) ────────────────────────────
    # Workers are identified by LSTM_NUM_WORKERS > 1 (product mode) or by
    # LSTM_SEED being set (seed mode).  The bare coordinator has neither.
    is_worker = num_workers > 1 or "LSTM_SEED" in os.environ
    if PARALLEL and not is_worker:
        if PARALLEL_MODE == "seed":
            _run_parallel_seeds(
                seeds=SEEDS, max_concurrent=MAX_CONCURRENT, no_merge=NO_MERGE,
            )
        else:  # "product"
            _run_parallel_products(
                num_workers=N_PRODUCT_WORKERS, max_concurrent=MAX_CONCURRENT,
                no_merge=NO_MERGE,
            )
        return

    # Seed-mode workers run a single seed; product-mode workers run all SEEDS.
    seeds_to_run = ([int(os.environ["LSTM_SEED"])] if "LSTM_SEED" in os.environ
                    else list(SEEDS))

    target_products = [(911753,6269)]
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_feather(DATA_PATH)
    
    if DATE_COL in df.index.names:
        df = df.reset_index(drop=True) if DATE_COL in df.columns else df.reset_index()
    if df.index.name == DATE_COL:
         df = df.reset_index(drop=True)
         
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, "item_id", "store_id"]).reset_index(drop=True)
    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS)
    
    if target_products is None:
        print(f"Loading top products from {TOP_DATA_PATH}...")
        top_df = pd.read_feather(TOP_DATA_PATH)
        target_products = (
            top_df[["item_id", "store_id"]]
            .drop_duplicates()
            .sort_values(["item_id", "store_id"])
            .apply(lambda r: (int(r["item_id"]), int(r["store_id"])), axis=1)
            .tolist()
        )
    products = df[df[['item_id', 'store_id']].apply(tuple, axis=1).isin(target_products)][['item_id', 'store_id']].drop_duplicates().values[:5]

    # Apply PRODUCTS_END so the last product(s) can be excluded from the
    # parallel run and handled sequentially later.
    if PRODUCTS_END is not None:
        products = products[:PRODUCTS_END]

    # ── Product sharding (parallelism across workers) ────────────────────
    # Each worker processes a disjoint slice so multiple processes run
    # simultaneously without duplicating work.  In seed mode (num_workers==1)
    # this is a no-op and every worker handles all products for its seed.
    n_products_total = len(products)
    products = products[worker_id::num_workers]
    print(f"Worker {worker_id}/{num_workers}: processing "
          f"{len(products)} / {n_products_total} products  |  seeds={seeds_to_run}")

    #strategies= ['selected_epochs']
    strategies = ['best_val', 'combined', 'expanding_window', 'sliding_window']
    results = []

    # ── Per-worker CSV suffix (avoids concurrent-write races) ────────────
    # Each worker writes to its own *_w{id}.csv; the coordinator merges them
    # into the canonical CSV after all workers finish.
    worker_csv_suffix = f"_w{worker_id}" if num_workers > 1 else ""
    RESULTS_CSV = os.path.join(
        SCRIPT_DIR, RESULTS_CSV_NAME.replace('.csv', f'{worker_csv_suffix}.csv')
    )

    os.makedirs('best_models', exist_ok=True)
    os.makedirs('grid_search_plots', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    criterion2 = nn.MSELoss()

    for seed in seeds_to_run:
        for item_id, store_id in products:
            df_product = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].copy()
            df_product[DATE_COL] = pd.to_datetime(df_product[DATE_COL])
            df_product = df_product.sort_values(DATE_COL).reset_index(drop=True)
            
            test_start_idx = len(df_product) - forecast_horizon
            val_start_idx = test_start_idx - val_size
            train_start_idx = val_start_idx - train_size

            train_slice = slice(train_start_idx, val_start_idx)
            val_slice = slice(val_start_idx, test_start_idx)
            test_slice = slice(test_start_idx, None)
            
            train = df_product[TARGET_COL][train_slice].values
            val = df_product[TARGET_COL][val_slice].values
            test = df_product[TARGET_COL][test_slice].values
            
            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
            val_scaled = scaler.transform(val.reshape(-1, 1)).flatten()

            exog_train = df_product[EXOG_COLS][train_slice].values
            exog_val = df_product[EXOG_COLS][val_slice].values
            exog_test = df_product[EXOG_COLS][test_slice].values
            
            exog_scaler = MinMaxScaler()
            exog_train_scaled = exog_scaler.fit_transform(exog_train)
            exog_val_scaled = exog_scaler.transform(exog_val)
            exog_test_scaled = exog_scaler.transform(exog_test)
            
            input_size = 1 + len(EXOG_COLS)
            
            train_dataset = TimeSeriesDataset(train_scaled, exog_train_scaled, lookback_window)
            val_dataset = TimeSeriesDataset(val_scaled, exog_val_scaled, lookback_window)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            
            combined_train_val = np.concatenate([train_scaled, val_scaled])
            combined_exog = np.concatenate([exog_train_scaled, exog_val_scaled]) if exog_train_scaled is not None else None
            combined_dataset = TimeSeriesDataset(combined_train_val, combined_exog, lookback_window)
            combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)

            strategy_forecasts = {}
            strategy_train_losses = {}
            strategy_val_losses = {}
            strategy_rmses = {}
            strategy_maes = {}
            strategy_biases = {}
            strategy_scores = {}
            strategy_pocids = {}

            train_index = df_product[DATE_COL][train_slice].values
            val_index = df_product[DATE_COL][val_slice].values
            test_index = df_product[DATE_COL][test_slice].values

            for strategy in strategies:
                print(f"\\n--- Strategy: {strategy}, Seed: {seed}, Item: {item_id}, Store: {store_id} ---")
                model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=150//3)
                
                model_dir = f'best_models/seed_{seed}/{loss_type}/{strategy}'
                os.makedirs(model_dir, exist_ok=True)
                model_path = f'{model_dir}/lstm_item{item_id}_store{store_id}.pth'
                
                if strategy == 'best_val':
                    model, t_losses, v_losses, best_epoch, train_time = train_model(
                        seed, EPOCHS, model, train_loader, val_loader, EXOG_COLS, 
                        criterion, criterion2, optimizer, device, model_path, scheduler, 150)
                elif strategy == 'combined':
                    # Save pristine state before train_model modifies model/optimizer
                    initial_model_state = copy.deepcopy(model.state_dict())

                    # Find optimal epoch count via val loss early stopping
                    model, t_init, v_losses, optimal_epoch, _ = train_model(
                        seed, EPOCHS, model, train_loader, val_loader, EXOG_COLS,
                        criterion, criterion2, optimizer, device, model_path + "_temp.pth", scheduler, 150)

                    # Reset to pristine weights and a fresh optimizer before refit
                    model.load_state_dict(initial_model_state)
                    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

                    # Refit from scratch on combined data for optimal_epoch epochs
                    model, t_losses, train_time = train_model_combined(
                        seed, optimal_epoch, model, combined_loader, criterion, optimizer, device, model_path)
                    best_epoch = optimal_epoch
                

                elif strategy == 'expanding_window':
                    model, t_losses, v_losses, best_epoch, train_time = train_model_expanding_window(
                        seed=seed, epochs=EPOCHS, model=model,
                        full_train_scaled=combined_train_val, exog_scaled=combined_exog,
                        seq_length=lookback_window, initial_train_size=cv_initial_train_size,
                        val_step_size=cv_val_step_size, batch_size=batch_size,
                        criterion=criterion, criterion2=criterion2, 
                        optimizer=optimizer, device=device, final_model_path=model_path, 
                        dataset_class=TimeSeriesDataset, scheduler=scheduler, patience=150
                    )
                elif strategy == 'sliding_window':
                    model, t_losses, v_losses, best_epoch, train_time = train_model_sliding_window(
                        seed=seed, epochs=EPOCHS, model=model,
                        full_train_scaled=combined_train_val, exog_scaled=combined_exog,
                        seq_length=lookback_window, initial_train_size=cv_initial_train_size,
                        val_step_size=cv_val_step_size, batch_size=batch_size,
                        criterion=criterion, criterion2=criterion2, 
                        optimizer=optimizer, device=device, final_model_path=model_path, 
                        dataset_class=TimeSeriesDataset, scheduler=scheduler, patience=150
                    )

                strategy_train_losses[strategy] = t_losses
                strategy_val_losses[strategy] = v_losses

                # Load best model
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path))
                
                # Inference
                forecast, infer_time = recursive_inference(
                    model, test_start_idx, lookback_window, val_scaled, exog_val_scaled, exog_test_scaled, exog_test,
                    scaler, exog_scaler, df_product, device, EXOG_COLS, forecast_horizon, seed, strategy, item_id, store_id, loss_type, SCRIPT_DIR
                )
                
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

                strategy_forecasts[strategy] = forecast
                strategy_rmses[strategy] = rmse
                strategy_maes[strategy] = mae
                strategy_biases[strategy] = bias
                strategy_scores[strategy] = score
                strategy_pocids[strategy] = pocid
                
                row = {
                    'seed': seed,
                    'strategy': strategy,
                    'item_id': item_id,
                    'store_id': store_id,
                    'rmse': rmse,
                    'mae': mae,
                    'train_time': train_time,
                    'inference_time': infer_time,
                    'best_epoch': best_epoch
                }
                results.append(row)
                pd.DataFrame([row]).to_csv(
                    RESULTS_CSV,
                    mode='a',
                    header=not os.path.exists(RESULTS_CSV),
                    index=False,
                )

            # --- Plot Forecast Comparisons ---
            plot_dir = os.path.join(SCRIPT_DIR, f'grid_search_plots/seed_{seed}/{loss_type}')
            os.makedirs(plot_dir, exist_ok=True)
            
            #strategies_str = "_".join(strategies)
            strategies_str = "_"
            plot_path = os.path.join(plot_dir, f'item{item_id}_store{store_id}_{strategies_str}_comparison.html')
            
            plot_results(train, val, test, strategy_forecasts, 
                         train_index, val_index, test_index,
                         strategy_train_losses, strategy_val_losses,
                         target_col=TARGET_COL, title=f'LSTM Strategies Comparison - Item {item_id} Store {store_id} (Seed {seed})',
                         save_path=plot_path,
                         rmse=strategy_rmses, mae=strategy_maes, bias=strategy_biases,
                         score=strategy_scores, pocid=strategy_pocids, df_full=df_product)

    print(f"Experiments completed. Results saved to {RESULTS_CSV}.")

if __name__ == "__main__":
    main()
