"""
Seasonal-Naive baseline (Hyndman & Athanasopoulos, FPP3 §5.2).

Protocol
--------
* Forecast for day ``T + h`` (h = 1..H) is the value observed one season earlier,
  repeating the last observed season:  ``y_hat[T+h] = y[T + h - m * (floor((h-1)/m) + 1)]``.
* ``m = SEASON_LENGTH`` (7 = weekly seasonality for daily sales).
* The history ends at the last validation day, so the 152-day test forecast starts
  from exactly the same point as the other baselines (no test data is used).
* No training, no scaling, no exogenous features, no randomness.

Seeds: the method is deterministic, so every seed gives the same forecast.  One row
per seed is still written so the results CSV lines up row-for-row with the other
baselines (same seeds x products) for the statistical comparisons.

Run from anywhere:  python Baselines/Seasonal-Naive/seasonal_naive_run.py
"""

import importlib.util
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

script_dir = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(script_dir, '..', '..'))


def _load_plot_results():
    """plot_results from GNN/plots.py, loaded by file path.  None if unavailable."""
    path = os.path.join(REPO_DIR, 'GNN', 'plots.py')
    try:
        spec = importlib.util.spec_from_file_location('baseline_plots', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.plot_results
    except Exception as exc:                       # missing file or plotly
        print(f"[plots disabled] {exc}")
        return None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join(REPO_DIR, 'dataset', 'data_andre.feather')
DATE_COL = 'date'
TARGET_COL = 'value'

# Same splits / products / seeds as the other baselines, so results are comparable.
train_size = 455
val_size = 154
forecast_horizon = 152
SEASON_LENGTH = 7        # m: weekly seasonality

seeds = [42, 1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]
target_products = [26008, 907969, 907967, 213626]
SAVE_PLOTS = True

MODEL_NAME = 'SeasonalNaive'
RESULTS_CSV = os.path.join(script_dir, 'seasonal_naive_results.csv')


def metrics(test, forecast):
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    bias = np.mean(forecast - test)
    d_true, d_pred = np.diff(test), np.diff(forecast)
    pocid = float(((d_true * d_pred) > 0).mean()) if len(d_true) else 0.0
    score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)
    return rmse, mae, bias, score, pocid


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def seasonal_naive_forecast(history, horizon, m=SEASON_LENGTH):
    """Repeat the last ``m`` observations of ``history`` for ``horizon`` steps."""
    history = np.asarray(history, dtype=float)
    if len(history) < m:
        raise ValueError(f"history ({len(history)}) shorter than season length ({m})")
    last_season = history[-m:]
    return last_season[np.arange(horizon) % m]


# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_feather(DATA_PATH)
    if DATE_COL in df.index.names:
        df = df.reset_index(drop=True) if DATE_COL in df.columns else df.reset_index()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, "item_id", "store_id"]).reset_index(drop=True)

    products = df[df['item_id'].isin(target_products)][['item_id', 'store_id']].drop_duplicates().values[:5]
    plot_results = _load_plot_results() if SAVE_PLOTS else None

    results = []
    for item_id, store_id in products:
        df_product = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].copy()
        df_product = df_product.sort_values(DATE_COL).reset_index(drop=True)

        test_start_idx = len(df_product) - forecast_horizon
        val_start_idx = test_start_idx - val_size
        train_start_idx = val_start_idx - train_size
        train_slice = slice(train_start_idx, val_start_idx)
        val_slice = slice(val_start_idx, test_start_idx)
        test_slice = slice(test_start_idx, None)

        series = df_product[TARGET_COL].values.astype(float)
        train, val, test = series[train_slice], series[val_slice], series[test_slice]

        # Sanity check on validation: forecast val from the end of train.
        val_fc = seasonal_naive_forecast(train, len(val))
        _, val_mae, _, _, _ = metrics(val, val_fc)

        # Test forecast from the end of validation (same origin as the other baselines).
        t0 = time.time()
        forecast = seasonal_naive_forecast(np.concatenate([train, val]), forecast_horizon)
        infer_time = time.time() - t0

        rmse, mae, bias, score, pocid = metrics(test, forecast)
        print(f"--- {MODEL_NAME} | Item: {item_id}, Store: {store_id} --- "
              f"RMSE {rmse:.4f} | MAE {mae:.4f} | bias {bias:.4f} | POCID {pocid:.3f} "
              f"| val MAE {val_mae:.4f}")

        for seed in seeds:
            results.append({
                'seed': seed, 'model': MODEL_NAME, 'item_id': item_id, 'store_id': store_id,
                'rmse': rmse, 'mae': mae, 'bias': bias, 'score': score, 'pocid': pocid,
                'train_time': 0.0, 'inference_time': infer_time, 'best_epoch': 0,
            })

        if plot_results is not None:
            plot_dir = os.path.join(script_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_results(
                train, val, test, {MODEL_NAME: forecast},
                df_product[DATE_COL][train_slice].values,
                df_product[DATE_COL][val_slice].values,
                df_product[DATE_COL][test_slice].values,
                {MODEL_NAME: []}, {MODEL_NAME: []}, target_col=TARGET_COL,
                title=f'{MODEL_NAME} (m={SEASON_LENGTH}) - Item {item_id} Store {store_id}',
                save_path=os.path.join(plot_dir, f'item{item_id}_store{store_id}.html'),
                rmse={MODEL_NAME: rmse}, mae={MODEL_NAME: mae}, bias={MODEL_NAME: bias},
                score={MODEL_NAME: score}, pocid={MODEL_NAME: pocid}, df_full=df_product)

    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    print(f"Experiments completed. Results saved to {RESULTS_CSV}.")


if __name__ == "__main__":
    main()
