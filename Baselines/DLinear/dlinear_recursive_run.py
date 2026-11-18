"""
DLinear with iterative (recursive) multi-step forecasting.

Same protocol as the LSTM / MLP baselines and the graph hybrids, so the
comparison is like-for-like (``dlinear_run.py`` keeps the original direct
LTSF-Linear protocol instead).

Protocol
--------
* One-step-ahead training: ``DLinear(horizon=1)`` maps the last ``LOOKBACK``
  days to the next day.
* Validation windows come from the validation slice only.
* AdamW + MSE + grad-clip 1.0 + ReduceLROnPlateau, early stopping that restores
  the best-validation weights (``Baselines/window_forecasting.py``).
* Test: recursive rollout seeded with the last ``LOOKBACK`` validation days;
  each prediction is fed back into the window for the next step (152 steps).
* MinMaxScaler on target and exog, fitted on the train split only (as the LSTM).

Exogenous variables (``USE_EXOG = True``)
-----------------------------------------
* Window row k carries ``[target_k | exog_{k+1}]``: the last row holds the
  calendar / holiday features of the day being predicted (``LSTM/dataset.py``
  convention).
* DLinear reads them through its covariate branch: one linear map over the
  ``LOOKBACK x n_exog`` block, zero-initialised, added to the trend + seasonal
  output (``DLinear(n_covariates=len(EXOG_COLS))``).
* Calendar / holiday features are known in advance, so their true test values
  are used.  ``lag_k`` / ``rolling_mean_excl_W`` columns depend on the target,
  so during the rollout they are recomputed from the history + the model's own
  forecasts (no test values leak).

Run from anywhere:  python Baselines/DLinear/dlinear_recursive_run.py
"""

import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.abspath(os.path.join(script_dir, '..'))
REPO_DIR = os.path.abspath(os.path.join(script_dir, '..', '..'))
for p in (script_dir, BASELINES_DIR, REPO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from dlinear import DLinear                                     # noqa: E402
from window_forecasting import fit_and_forecast, reseed         # noqa: E402
from model_utils.utils import generate_exogenous_features       # noqa: E402


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
LOOKBACK = 30            # same lookback as the LSTM baseline (>= max lag, so lag_30 is exact)
KERNEL_SIZE = 7          # moving-average window for the trend (odd, <= LOOKBACK)

batch_size = 32
EPOCHS = 1000
PATIENCE = 100
LEARNING_RATE = 0.001
USE_SCHEDULER = True

USE_EXOG = True
# Same exogenous set as Baselines/LSTM/lstm_run.py.
EXOG_COLS = [
    "day_of_week", "day_of_month", "week_of_year", "week_of_month",
    "month", "quarter", "is_weekend",
    "lag_1", "lag_7", "lag_30",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_monday", "is_friday",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_pre_holiday_1", "is_pre_holiday_2", "is_pre_holiday_3", "is_pre_holiday_7",
    "is_post_holiday_1", "is_post_holiday_2", "is_post_holiday_3", "is_post_holiday_7",
    "is_bridge_day",
]

seeds = [42, 1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]
target_products = [26008, 907969, 907967, 213626]
SAVE_PLOTS = True

MODEL_NAME = 'DLinear-Recursive-Exog' if USE_EXOG else 'DLinear-Recursive'
RESULTS_CSV = os.path.join(
    script_dir, 'dlinear_recursive_exog_results.csv' if USE_EXOG else 'dlinear_recursive_results.csv')


def metrics(test, forecast):
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    bias = np.mean(forecast - test)
    d_true, d_pred = np.diff(test), np.diff(forecast)
    pocid = float(((d_true * d_pred) > 0).mean()) if len(d_true) else 0.0
    score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)
    return rmse, mae, bias, score, pocid


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
    exog_cols = EXOG_COLS if USE_EXOG else []
    if exog_cols:
        df = generate_exogenous_features(df, date_col=DATE_COL, target_col=TARGET_COL,
                                         exog_cols=exog_cols)

    products = df[df['item_id'].isin(target_products)][['item_id', 'store_id']].drop_duplicates().values[:5]
    plot_results = _load_plot_results() if SAVE_PLOTS else None

    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for seed in seeds:
        for item_id, store_id in products:
            print(f"\n--- {MODEL_NAME} | Seed: {seed}, Item: {item_id}, Store: {store_id} ---")
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

            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train.reshape(-1, 1)).ravel()
            val_scaled = scaler.transform(val.reshape(-1, 1)).ravel()

            exog_kwargs = {}
            if exog_cols:
                exog = df_product[exog_cols].values.astype(float)
                exog_scaler = MinMaxScaler()
                exog_kwargs = dict(
                    exog_train=exog_scaler.fit_transform(exog[train_slice]),
                    exog_val=exog_scaler.transform(exog[val_slice]),
                    exog_test=exog_scaler.transform(exog[test_slice]),
                    exog_scaler=exog_scaler, exog_cols=exog_cols,
                    # real history up to the forecast origin, for lag / rolling recomputation
                    target_history_unscaled=np.concatenate([train, val]),
                )

            reseed(seed)
            model = DLinear(LOOKBACK, horizon=1, kernel_size=KERNEL_SIZE,
                            n_covariates=len(exog_cols))

            t0 = time.time()
            forecast, t_losses, v_losses = fit_and_forecast(
                model, lookback=LOOKBACK, train_scaled=train_scaled, val_scaled=val_scaled,
                scaler=scaler, horizon=forecast_horizon, device=device, epochs=EPOCHS,
                patience=PATIENCE, batch_size=batch_size, lr=LEARNING_RATE,
                use_scheduler=USE_SCHEDULER, **exog_kwargs)
            total_time = time.time() - t0
            best_epoch = int(np.argmin(v_losses)) + 1 if v_losses else 0

            model_dir = os.path.join(script_dir, 'best_models_recursive', f'seed_{seed}')
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(model_dir, f'dlinear_item{item_id}_store{store_id}.pth'))

            rmse, mae, bias, score, pocid = metrics(test, forecast)
            print(f"epochs {len(t_losses)} (best {best_epoch}) | RMSE {rmse:.4f} | MAE {mae:.4f} "
                  f"| bias {bias:.4f} | POCID {pocid:.3f}")

            results.append({
                'seed': seed, 'model': MODEL_NAME, 'item_id': item_id, 'store_id': store_id,
                'rmse': rmse, 'mae': mae, 'bias': bias, 'score': score, 'pocid': pocid,
                'train_time': total_time, 'inference_time': np.nan, 'best_epoch': best_epoch,
            })
            # Save after every model so an interrupted run keeps its results.
            pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)

            if plot_results is not None:
                plot_dir = os.path.join(script_dir, 'plots_recursive', f'seed_{seed}')
                os.makedirs(plot_dir, exist_ok=True)
                plot_results(
                    train, val, test, {MODEL_NAME: forecast},
                    df_product[DATE_COL][train_slice].values,
                    df_product[DATE_COL][val_slice].values,
                    df_product[DATE_COL][test_slice].values,
                    {MODEL_NAME: t_losses}, {MODEL_NAME: v_losses}, target_col=TARGET_COL,
                    title=f'{MODEL_NAME} - Item {item_id} Store {store_id} (Seed {seed})',
                    save_path=os.path.join(plot_dir, f'item{item_id}_store{store_id}.html'),
                    rmse={MODEL_NAME: rmse}, mae={MODEL_NAME: mae}, bias={MODEL_NAME: bias},
                    score={MODEL_NAME: score}, pocid={MODEL_NAME: pocid}, df_full=df_product)

    print(f"Experiments completed. Results saved to {RESULTS_CSV}.")


if __name__ == "__main__":
    main()
