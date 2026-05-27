"""
Run LSTM forecasting for every representative product and record all per-product
metrics in a single CSV file (for dissertation reporting).

Adapted from lstm_strategies.py. Differences:
  * Iterates over all (item_id, store_id) pairs in
    dataset/representative_items.feather instead of a handful of target products.
  * Writes results incrementally so the run can be resumed safely after a crash.
  * Plotting is OFF by default (plotting ~1000 products is impractical); enable
    it by setting MAKE_PLOTS = True.
  * Defaults to a single seed and a single strategy to keep total runtime
    bounded. Add more if needed.
"""

import sys
import os
import time
import traceback

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
from train import (
    train_model,
    train_model_best_train_loss,
    train_model_combined,
    train_model_expanding_window,
    train_model_sliding_window,
)
from inference import recursive_inference

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, '..', 'dataset', 'data_andre.feather')
REPRESENTATIVE_PATH = os.path.join(
    script_dir, '..', 'dataset', 'representative_items.feather'
)
RESULTS_DIR = script_dir  # per-seed CSVs are written here

DATE_COL = 'date'
TARGET_COL = 'value'

# train_size is computed dynamically per product as: len(df_product) - val_size - forecast_horizon
val_size = 154
forecast_horizon = 152
lookback_window = 7

EXOG_COLS = [
    "day_of_week", "day_of_month", "week_of_year", "week_of_month",
    "month", "quarter", "is_weekend",
    "lag_1", "lag_7",  "lag_30",
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
PATIENCE = 150

# Keep these small by default - running every strategy x every seed x every
# product is not feasible. Add more entries if you have the time budget.
SEEDS = [42, 1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]
STRATEGIES = ['best_val']  # any of: best_val, best_train_early_val, combined,
                           # expanding_window, sliding_window
LOSS_TYPE = 'MSELoss'

MAKE_PLOTS = True            # set True to also produce per-product HTML plots
SAVE_EVERY = 1               # flush CSV every N products processed
RESUME = True                # skip (seed, strategy, item, store) already in CSV


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_representative_products() -> pd.DataFrame:
    """Return unique (item_id, store_id) pairs from representative_items.feather."""
    rep = pd.read_feather(REPRESENTATIVE_PATH)
    pairs = (
        rep[['item_id', 'store_id']]
        .drop_duplicates()
        .sort_values(['item_id', 'store_id'])
        .reset_index(drop=True)
    )
    return pairs


def get_seed_csv(seed: int) -> str:
    """Return the path of the per-seed results CSV."""
    return os.path.join(RESULTS_DIR, f'lstm_results_seed_{seed}.csv')


def load_existing_results(seed: int) -> set:
    """Return a set of (seed, strategy, item_id, store_id) tuples already done for this seed."""
    csv_path = get_seed_csv(seed)
    if not (RESUME and os.path.exists(csv_path)):
        return set()
    try:
        prev = pd.read_csv(csv_path)
    except Exception:
        return set()
    needed = {'seed', 'strategy', 'item_id', 'store_id'}
    if not needed.issubset(prev.columns):
        return set()
    return set(
        zip(
            prev['seed'].astype(int),
            prev['strategy'].astype(str),
            prev['item_id'].astype(int),
            prev['store_id'].astype(int),
        )
    )


def append_result(row: dict, csv_path: str) -> None:
    """Append a single result row to the per-seed CSV, creating it if needed."""
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode='a', header=write_header, index=False)


def build_best_seed_csv() -> None:
    """Merge all per-seed CSVs and write one row per (item_id, store_id, strategy)
    selecting the seed with the lowest composite score."""
    frames = []
    for seed in SEEDS:
        path = get_seed_csv(seed)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        print("[WARN] No per-seed CSVs found; skipping best-seed summary.")
        return
    combined = pd.concat(frames, ignore_index=True)
    ok = combined[combined['status'] == 'ok'].copy()
    if ok.empty:
        print("[WARN] No successful rows found; skipping best-seed summary.")
        return
    best_idx = ok.groupby(['item_id', 'store_id', 'strategy'])['score'].idxmin()
    best = ok.loc[best_idx].reset_index(drop=True)
    best_path = os.path.join(RESULTS_DIR, 'lstm_best_seed_results.csv')
    best.to_csv(best_path, index=False)
    print(f"Best-seed summary saved to {best_path} ({len(best)} rows).")


def run_strategy(
    strategy, seed, model, train_loader, val_loader, combined_loader,
    combined_train_val, combined_exog, criterion, criterion2, optimizer,
    scheduler, device, model_path, train_size,
):
    """Train one strategy and return (model, t_losses, v_losses, best_epoch, train_time)."""
    if strategy == 'best_val':
        return train_model(
            seed, EPOCHS, model, train_loader, val_loader, EXOG_COLS,
            criterion, criterion2, optimizer, device, model_path, scheduler, PATIENCE,
        )
    if strategy == 'best_train_early_val':
        return train_model_best_train_loss(
            seed, EPOCHS, model, train_loader, val_loader, EXOG_COLS,
            criterion, criterion2, optimizer, device, model_path, scheduler, PATIENCE,
        )
    if strategy == 'combined':
        model, _, v_losses, optimal_epoch, _ = train_model(
            seed, EPOCHS, model, train_loader, val_loader, EXOG_COLS,
            criterion, criterion2, optimizer, device,
            model_path + "_temp.pth", scheduler, PATIENCE,
        )
        if os.path.exists(model_path + "_temp.pth"):
            model.load_state_dict(torch.load(model_path + "_temp.pth"))
        model, t_losses, train_time = train_model_combined(
            seed, optimal_epoch, model, combined_loader, criterion,
            optimizer, device, model_path,
        )
        return model, t_losses, v_losses, optimal_epoch, train_time
    if strategy == 'expanding_window':
        return train_model_expanding_window(
            seed=seed, epochs=EPOCHS, model=model,
            full_train_scaled=combined_train_val, exog_scaled=combined_exog,
            seq_length=lookback_window, initial_train_size=train_size,
            val_step_size=30, batch_size=batch_size,
            criterion=criterion, criterion2=criterion2,
            optimizer=optimizer, device=device, final_model_path=model_path,
            dataset_class=TimeSeriesDataset, scheduler=scheduler, patience=PATIENCE,
        )
    if strategy == 'sliding_window':
        return train_model_sliding_window(
            seed=seed, epochs=EPOCHS, model=model,
            full_train_scaled=combined_train_val, exog_scaled=combined_exog,
            seq_length=lookback_window, initial_train_size=train_size,
            val_step_size=30, batch_size=batch_size,
            criterion=criterion, criterion2=criterion2,
            optimizer=optimizer, device=device, final_model_path=model_path,
            dataset_class=TimeSeriesDataset, scheduler=scheduler, patience=PATIENCE,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def impute_product_dates(
    df_product: pd.DataFrame,
    full_date_range: pd.DatetimeIndex,
    item_id: int,
    store_id: int,
) -> pd.DataFrame:
    """Reindex a product's DataFrame to cover every date in full_date_range.

    Missing dates get value=0.  Exogenous features are recomputed from scratch
    for the padded series so that lags reflect the imputed 0s correctly.
    """
    df_product = df_product.copy()
    df_product[DATE_COL] = pd.to_datetime(df_product[DATE_COL])
    df_product = df_product.set_index(DATE_COL)
    df_product = df_product.reindex(full_date_range)
    df_product.index.name = DATE_COL

    # Fill identity columns and sales
    df_product['item_id'] = item_id
    df_product['store_id'] = store_id
    df_product[TARGET_COL] = df_product[TARGET_COL].fillna(0.0)

    # Drop stale exog columns — they will be regenerated below
    cols_to_drop = [c for c in EXOG_COLS if c in df_product.columns]
    df_product = df_product.drop(columns=cols_to_drop)

    df_product = df_product.reset_index()  # date becomes a regular column again
    df_product = generate_exogenous_features(
        df_product, date_col=DATE_COL, exog_cols=EXOG_COLS
    )
    return df_product.sort_values(DATE_COL).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_feather(DATA_PATH)

    if DATE_COL in df.index.names:
        df = df.reset_index(drop=True) if DATE_COL in df.columns else df.reset_index()
    if df.index.name == DATE_COL:
        df = df.reset_index(drop=True)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, "item_id", "store_id"]).reset_index(drop=True)
    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS)

    full_date_range = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq='D')
    print(f"Full date range: {full_date_range[0].date()} → {full_date_range[-1].date()} ({len(full_date_range)} days)")

    products = load_representative_products()
    print(f"Found {len(products)} representative (item_id, store_id) pairs.")

    os.makedirs('best_models', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    criterion = nn.MSELoss()
    criterion2 = nn.MSELoss()

    total_combos = len(SEEDS) * len(STRATEGIES) * len(products)
    print(f"Total (seed x strategy x product) combinations: {total_combos}")

    overall_start = time.time()
    done_counter = 0

    for seed in SEEDS:
        seed_csv = get_seed_csv(seed)
        already_done = load_existing_results(seed)
        if already_done:
            print(f"[seed={seed}] Resuming: {len(already_done)} rows already in {seed_csv}.")

        for prod_idx, (item_id, store_id) in enumerate(
            products.itertuples(index=False, name=None), start=1
        ):
            item_id = int(item_id)
            store_id = int(store_id)

            df_product = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].copy()

            n_raw = len(df_product)
            df_product = impute_product_dates(df_product, full_date_range, item_id, store_id)
            if len(df_product) > n_raw:
                print(
                    f"  [IMPUTE] item={item_id} store={store_id}: "
                    f"{len(df_product) - n_raw} missing days filled with 0"
                )

            # Dynamic train size: use all available rows before val and test
            train_size = len(df_product) - val_size - forecast_horizon
            min_train = lookback_window + 1

            if train_size < min_train:
                print(
                    f"[SKIP] item={item_id} store={store_id}: only {len(df_product)} rows, "
                    f"need at least {min_train + val_size + forecast_horizon} "
                    f"(lookback+1 + val + forecast)."
                )
                for strategy in STRATEGIES:
                    append_result({
                        'seed': seed, 'strategy': strategy,
                        'item_id': item_id, 'store_id': store_id,
                        'rmse': np.nan, 'mae': np.nan, 'bias': np.nan,
                        'pocid': np.nan, 'score': np.nan,
                        'train_time': np.nan, 'inference_time': np.nan,
                        'best_epoch': np.nan, 'n_rows': len(df_product),
                        'status': 'skipped_insufficient_data',
                    }, seed_csv)
                continue

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
            combined_exog = (
                np.concatenate([exog_train_scaled, exog_val_scaled])
                if exog_train_scaled is not None else None
            )
            combined_dataset = TimeSeriesDataset(
                combined_train_val, combined_exog, lookback_window
            )
            combined_loader = DataLoader(
                combined_dataset, batch_size=batch_size, shuffle=False
            )

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

            for strategy in STRATEGIES:
                key = (seed, strategy, item_id, store_id)
                if key in already_done:
                    print(
                        f"[{prod_idx}/{len(products)}] (seed={seed}, {strategy}) "
                        f"item={item_id} store={store_id}: already done, skipping."
                    )
                    continue

                print(
                    f"\n[{prod_idx}/{len(products)}] seed={seed} strategy={strategy} "
                    f"item={item_id} store={store_id}"
                )

                try:
                    model = LSTM(
                        input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, dropout=dropout,
                    ).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=PATIENCE // 3,
                    )

                    model_dir = os.path.join(
                        script_dir, 'best_models', f'seed_{seed}',
                        LOSS_TYPE, strategy,
                    )
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(
                        model_dir, f'lstm_item{item_id}_store{store_id}.pth'
                    )

                    model, t_losses, v_losses, best_epoch, train_time = run_strategy(
                        strategy, seed, model, train_loader, val_loader,
                        combined_loader, combined_train_val, combined_exog,
                        criterion, criterion2, optimizer, scheduler, device,
                        model_path, train_size,
                    )

                    strategy_train_losses[strategy] = t_losses
                    strategy_val_losses[strategy] = v_losses

                    if os.path.exists(model_path):
                        model.load_state_dict(torch.load(model_path))

                    forecast, infer_time = recursive_inference(
                        model, test_start_idx, lookback_window, val_scaled,
                        exog_val_scaled, exog_test_scaled, exog_test,
                        scaler, exog_scaler, df_product, device, EXOG_COLS,
                        forecast_horizon, seed, strategy, item_id, store_id,
                        LOSS_TYPE, script_dir,
                    )

                    rmse = float(np.sqrt(mean_squared_error(test, forecast)))
                    mae = float(mean_absolute_error(test, forecast))
                    bias = float(np.mean(forecast - test))

                    diff_original = test[1:] - test[:-1]
                    diff_pred = forecast[1:] - forecast[:-1]
                    is_positive = (diff_original * diff_pred) > 0
                    pocid = (
                        float(is_positive.sum() / len(is_positive))
                        if len(is_positive) > 0 else 0.0
                    )

                    score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)

                    strategy_forecasts[strategy] = forecast
                    strategy_rmses[strategy] = rmse
                    strategy_maes[strategy] = mae
                    strategy_biases[strategy] = bias
                    strategy_scores[strategy] = score
                    strategy_pocids[strategy] = pocid

                    append_result({
                        'seed': seed,
                        'strategy': strategy,
                        'item_id': item_id,
                        'store_id': store_id,
                        'rmse': rmse,
                        'mae': mae,
                        'bias': bias,
                        'pocid': pocid,
                        'score': score,
                        'train_time': train_time,
                        'inference_time': infer_time,
                        'best_epoch': best_epoch,
                        'n_rows': len(df_product),
                        'status': 'ok',
                    }, seed_csv)
                    already_done.add(key)
                    done_counter += 1

                    if done_counter % SAVE_EVERY == 0:
                        elapsed = time.time() - overall_start
                        print(
                            f"   -> rmse={rmse:.4f} mae={mae:.4f} bias={bias:.4f} "
                            f"pocid={pocid:.3f} | elapsed={elapsed/60:.1f}min"
                        )

                except Exception as exc:
                    print(
                        f"[ERROR] seed={seed} strategy={strategy} "
                        f"item={item_id} store={store_id}: {exc}"
                    )
                    traceback.print_exc()
                    append_result({
                        'seed': seed,
                        'strategy': strategy,
                        'item_id': item_id,
                        'store_id': store_id,
                        'rmse': np.nan, 'mae': np.nan, 'bias': np.nan,
                        'pocid': np.nan, 'score': np.nan,
                        'train_time': np.nan, 'inference_time': np.nan,
                        'best_epoch': np.nan,
                        'n_rows': len(df_product),
                        'status': f'error: {type(exc).__name__}',
                    }, seed_csv)

            if MAKE_PLOTS and strategy_forecasts:
                try:
                    plot_dir = os.path.join(
                        script_dir, 'LSTM_Plot_Results', str(item_id)
                    )
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_path = os.path.join(
                        plot_dir,
                        f'item{item_id}_store{store_id}_all_strategies_comparison.html',
                    )
                    plot_results(
                        train, val, test, strategy_forecasts,
                        train_index, val_index, test_index,
                        strategy_train_losses, strategy_val_losses,
                        target_col=TARGET_COL,
                        title=(
                            f'LSTM Comparison - Item {item_id} Store {store_id} '
                            f'(Seed {seed})'
                        ),
                        save_path=plot_path,
                        rmse=strategy_rmses, mae=strategy_maes,
                        bias=strategy_biases, score=strategy_scores,
                        pocid=strategy_pocids, df_full=df_product,
                    )
                except Exception as exc:
                    print(f"[WARN] plot failed for item={item_id} store={store_id}: {exc}")

    total_elapsed = time.time() - overall_start
    print(f"\nAll seeds done. Total runtime: {total_elapsed/60:.1f} min.")

    build_best_seed_csv()


if __name__ == "__main__":
    main()
