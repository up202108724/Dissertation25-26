"""
SARIMAX runner — iterates over every product-store pair in top_12500.feather,
runs the standard full-history SARIMAX approach, extracts RMSE / MAE / R2 / bias / POCID, 
saves an interactive HTML plot per product, and appends all metrics to sarimax_results.csv.

Resume-safe: already-processed pairs are skipped on re-run.
"""

import os
import sys
import csv
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from utils import generate_exogenous_features
# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, '..')))

# Assuming you saved the streamlined class in 'sarimax.py'
from sarimax import SARIMAXPipeline
from plots import plot_results
from model_utils.utils import compute_metrics    # RMSE, MAE, bias, score, POCID

# ── Data paths ─────────────────────────────────────────────────────────────
FULL_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../dataset/data_andre_classified.feather'))
TOP_DATA_PATH  = os.path.normpath(os.path.join(SCRIPT_DIR, '../dataset/top_12500.feather'))
DATE_COL   = 'date'
TARGET_COL = 'value'

# ── Split sizes (must match the rest of the pipeline) ──────────────────────
VAL_SIZE        = 30
FORECAST_WINDOW = 155
TRAIN_SIZE      = 761 - VAL_SIZE - FORECAST_WINDOW  # 576

# ── ARIMA settings ─────────────────────────────────────────────────────────
MAXITER = 200   # SARIMAX optimiser max iterations
M_CANDIDATES = [1, 7, 14, 30]   # non-seasonal, weekly, biweekly, monthly — AIC picks the winner per product

# ── Output ─────────────────────────────────────────────────────────────────
SAVE_PLOTS  = True
PLOTS_DIR   = os.path.join(SCRIPT_DIR, 'sarimax_plots')
RESULTS_CSV = os.path.join(SCRIPT_DIR, 'sarimax_results.csv')

# Streamlined fieldnames (no lookback)
FIELDNAMES = [
    'item_id', 'store_id', 'best_m',
    'rmse', 'mae', 'r2', 'bias', 'pocid', 'composite_score',
    'order', 'seasonal_order', 'elapsed_s',
]

EXOG_COLS = [
    "is_month_start", "is_month_end", 
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_pre_holiday_1", "is_pre_holiday_2", "is_pre_holiday_3", "is_pre_holiday_7",
    "is_post_holiday_1", "is_post_holiday_2", "is_post_holiday_3", "is_post_holiday_7",
    "is_bridge_day",
]

# ──────────────────────────────────────────────────────────────────────────
def main():
    warnings.filterwarnings('ignore')

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading data from {TOP_DATA_PATH}...")
    df = pd.read_feather(TOP_DATA_PATH)
    # date may be stored as the index — promote it to a regular column
    if df.index.name == DATE_COL:
        df = df.reset_index()
    elif DATE_COL not in df.columns:
        raise ValueError(f"Cannot find '{DATE_COL}' in columns or index of {TOP_DATA_PATH}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, 'item_id', 'store_id']).reset_index(drop=True)
    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS)
    full_df = df.copy()
    products = full_df[['item_id', 'store_id']].drop_duplicates().reset_index(drop=True)
    n_total  = len(products)
    print(f"Running SARIMAX for {n_total} product-store pairs.")

    pipeline = SARIMAXPipeline(target_col=TARGET_COL, date_col=DATE_COL)

    # ── Resume: load already-processed pairs ──────────────────────────────
    done_set = set()
    if os.path.exists(RESULTS_CSV):
        done_df = pd.read_csv(RESULTS_CSV, dtype=str)
        for _, row in done_df.iterrows():
            done_set.add((str(row['item_id']), str(row['store_id'])))
        print(f"Resuming: {len(done_set)} products already processed.")

    if SAVE_PLOTS:
        os.makedirs(PLOTS_DIR, exist_ok=True)

    start_time = time.time()

    for i, prod_row in products.iterrows():
        item_id  = prod_row['item_id']
        store_id = prod_row['store_id']

        if (str(item_id), str(store_id)) in done_set:
            print(f"[{i+1}/{n_total}] Skipping item {item_id}, store {store_id} (already done)")
            continue

        print(f"\n[{i+1}/{n_total}] {'='*55}")
        print(f"Processing Item {item_id}, Store {store_id}")
        print('=' * 60)

        # ── Check sufficient rows ──────────────────────────────────────────
        df_p = (
            df[(df['item_id'] == item_id) & (df['store_id'] == store_id)]
            .sort_values(DATE_COL).reset_index(drop=True)
        )
        required = FORECAST_WINDOW + VAL_SIZE + TRAIN_SIZE
        if len(df_p) < required:
            print(f"  Skipping: {len(df_p)} rows < {required} required")
            continue

        n           = len(df_p)
        train_end   = n - (VAL_SIZE + FORECAST_WINDOW)
        val_end     = n - FORECAST_WINDOW
        train_start = train_end - TRAIN_SIZE

        train_index = df_p[DATE_COL].iloc[train_start:train_end].values
        val_index   = df_p[DATE_COL].iloc[train_end:val_end].values
        test_index  = df_p[DATE_COL].iloc[val_end:].values
        y_test      = df_p[TARGET_COL].iloc[val_end:].values

        # ── Seasonal period detection ──────────────────────────────────────
        print(f"  Seasonal candidates: {M_CANDIDATES} (AIC will select best per product)")

        row_data = {'item_id': item_id, 'store_id': store_id, 'best_m': None}

        # ── Approach: Standard SARIMAX ─────────────────────────────────────
        try:
            t0 = time.time()

            (rmse, mae, bias, composite_score,
             order, seasonal_order, forecast,
             y_train_val, y_test_checked) = pipeline.fit_forecast(
                df, item_id, store_id,
                train_size=TRAIN_SIZE, val_size=VAL_SIZE,
                forecast_window=FORECAST_WINDOW,
                m_candidates=M_CANDIDATES,
                exog_cols=EXOG_COLS, maxiter=MAXITER,
            )
            elapsed_s = round(time.time() - t0, 2)
            row_data['best_m'] = seasonal_order[3] if seasonal_order[3] > 1 else 1

            r2 = float(r2_score(y_test, forecast))
            _, _, _, _, pocid = compute_metrics(y_test, forecast)

            print(f"  Standard ({elapsed_s:.1f}s) → RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, Bias: {bias:.4f}, POCID: {pocid:.4f}")

            row_data.update({
                'rmse': rmse, 'mae': mae, 'r2': r2,
                'bias': bias, 'pocid': pocid, 'composite_score': composite_score,
                'order': str(order),
                'seasonal_order': str(seasonal_order),
                'elapsed_s': elapsed_s,
            })

            if SAVE_PLOTS:
                plot_results(
                    train=df_p[TARGET_COL].iloc[train_start:train_end].values,
                    val=df_p[TARGET_COL].iloc[train_end:val_end].values,
                    test=y_test,
                    forecast=forecast,
                    train_index=train_index,
                    val_index=val_index,
                    test_index=test_index,
                    train_losses=None,
                    val_losses=None,
                    title=(
                        f"SARIMAX Standard — Item {item_id}, Store {store_id} "
                        f"| m={row_data['best_m']} | Order={order}"
                    ),
                    rmse=rmse, mae=mae, bias=bias,
                    score=r2, pocid=pocid,
                    save_path=os.path.join(
                        PLOTS_DIR,
                        f"item{item_id}_store{store_id}_standard.html",
                    ),
                )

        except Exception as e:
            print(f"  Standard Pipeline FAILED: {e}")
            row_data.update({k: '' for k in [
                'rmse', 'mae', 'r2', 'bias', 'pocid', 'composite_score',
                'order', 'seasonal_order', 'elapsed_s',
            ]})

        # ── Append row to CSV (one write per product = crash-safe) ─────────
        file_exists = os.path.exists(RESULTS_CSV)
        with open(RESULTS_CSV, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

    elapsed = time.time() - start_time
    print(f"\nAll done. Total time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    print(f"Results saved to: {RESULTS_CSV}")

if __name__ == '__main__':
    main()