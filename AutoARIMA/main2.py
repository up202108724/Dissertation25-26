"""
AutoARIMA runner — iterates over every product-store pair in top_12500.feather,
runs both the lookback (rolling state propagation) and full-history SARIMAX
approaches, extracts RMSE / MAE / R2 / bias / POCID, saves an interactive HTML
plot per product, and appends all metrics to autoarima_results.csv.

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

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, '..')))

from autoarima import AutoARIMAPipeline          # autoarima.py (no val phase)
from plots import plot_results
from model_utils.utils import compute_metrics    # RMSE, MAE, bias, score, POCID

# ── Data paths ─────────────────────────────────────────────────────────────
FULL_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '../dataset/data_andre_classified.feather'))
TOP_DATA_PATH  = os.path.normpath(os.path.join(SCRIPT_DIR, '../dataset/top_12500.feather'))
DATE_COL   = 'date'
TARGET_COL = 'value'

# ── Split sizes (must match the rest of the pipeline) ──────────────────────

VAL_SIZE        = 20
FORECAST_WINDOW = 153
TRAIN_SIZE      = 761- VAL_SIZE - FORECAST_WINDOW  # 588
LOOKBACK_WINDOW = 30

# ── ARIMA settings ─────────────────────────────────────────────────────────
MAXITER = 200   # SARIMAX optimiser max iterations

# ── Output ─────────────────────────────────────────────────────────────────
SAVE_PLOTS  = True
PLOTS_DIR   = os.path.join(SCRIPT_DIR, 'autoarima_plots')
RESULTS_CSV = os.path.join(SCRIPT_DIR, 'autoarima_results.csv')

FIELDNAMES = [
    'item_id', 'store_id', 'best_m',
    'lb_rmse', 'lb_mae', 'lb_r2', 'lb_bias', 'lb_pocid',
    'lb_order', 'lb_seasonal_order',
    'full_rmse', 'full_mae', 'full_r2', 'full_bias', 'full_pocid',
    'full_order', 'full_seasonal_order',
]

EXOG_COLS = [
    # Calendar features ONLY if you disable internal SARIMAX seasonality (m=1)
    # "day_of_week", "month", "is_weekend", 
    
    # Keep all deterministic events
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

    # ── Load full time-series data ─────────────────────────────────────────
    print(f"Loading data from {FULL_DATA_PATH}...")
    df = pd.read_feather(FULL_DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, 'item_id', 'store_id']).reset_index(drop=True)

    # ── Load product list ──────────────────────────────────────────────────
    print(f"Loading product list from {TOP_DATA_PATH}...")
    top_df   = pd.read_feather(TOP_DATA_PATH)
    products = top_df[['item_id', 'store_id']].drop_duplicates().reset_index(drop=True)
    n_total  = len(products)
    print(f"Running AutoARIMA for {n_total} product-store pairs.")

    pipeline = AutoARIMAPipeline(target_col=TARGET_COL, date_col=DATE_COL)

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
        m_candidates = pipeline.get_seasonal_candidates(df, item_id, store_id)
        best_m   = m_candidates[-1] if len(m_candidates) > 1 else 1
        seasonal = best_m > 1
        print(f"  Seasonal candidates: {m_candidates} → using m={best_m}")

        row_data = {'item_id': item_id, 'store_id': store_id, 'best_m': best_m}

        # ── Approach 2: Full history ───────────────────────────────────────
        try:
            t0 = time.time()
            (full_rmse, full_mae,
             full_order, full_seasonal_order,
             full_forecast) = pipeline.fit_forecast_full(
                df, item_id, store_id,
                train_size=TRAIN_SIZE, val_size=VAL_SIZE,
                forecast_window=FORECAST_WINDOW,
                seasonal=seasonal, m=best_m, maxiter=MAXITER,
            )
            full_r2   = float(r2_score(y_test, full_forecast))
            full_bias = float(np.mean(full_forecast - y_test))
            _, _, _, _, full_pocid = compute_metrics(y_test, full_forecast)
            print(f"  Full     ({time.time()-t0:.1f}s) → RMSE: {full_rmse:.4f}, MAE: {full_mae:.4f}, R2: {full_r2:.4f}, POCID: {full_pocid:.4f}")

            row_data.update({
                'full_rmse': full_rmse, 'full_mae': full_mae,
                'full_r2': full_r2, 'full_bias': full_bias, 'full_pocid': full_pocid,
                'full_order': str(full_order),
                'full_seasonal_order': str(full_seasonal_order),
            })

            if SAVE_PLOTS:
                plot_results(
                    train=df_p[TARGET_COL].iloc[train_start:train_end].values,
                    val=df_p[TARGET_COL].iloc[train_end:val_end].values,
                    test=y_test,
                    forecast=full_forecast,
                    train_index=train_index,
                    val_index=val_index,
                    test_index=test_index,
                    train_losses=None,
                    val_losses=None,
                    title=(
                        f"AutoARIMA Full — Item {item_id}, Store {store_id} "
                        f"| m={best_m} | Order={full_order}"
                    ),
                    rmse=full_rmse, mae=full_mae, bias=full_bias,
                    score=full_r2, pocid=full_pocid,
                    save_path=os.path.join(
                        PLOTS_DIR,
                        f"item{item_id}_store{store_id}_full.html",
                    ),
                )

        except Exception as e:
            print(f"  Full FAILED: {e}")
            row_data.update({k: '' for k in [
                'full_rmse', 'full_mae', 'full_r2', 'full_bias', 'full_pocid',
                'full_order', 'full_seasonal_order',
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
