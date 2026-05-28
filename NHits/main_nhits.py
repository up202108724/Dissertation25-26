"""
NHITS strategy runner — direct multi-horizon forecasting per (item, store) pair.

Mirrors the purpose of ``NHits/main.py`` (the MLP recursive runner) but uses
N-HiTS: instead of predicting one step ahead and recursing for ``H`` steps,
each training window maps a length-L lookback to a length-H forecast in a
single forward pass.  Known future exogenous features (calendar / holidays)
are passed alongside the lookback.

Data layout, scaler convention, per-product loop, plotting and CSV results
follow ``main.py`` for direct comparability.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

script_dir = os.path.dirname(os.path.abspath(__file__))

from plots import plot_results
from utils import generate_exogenous_features
from dataset import make_direct_windows
from train import train_nhits
from nhits import NHITS


# ─────────────────────────────────────────────────────────────────────────
# Configuration (kept parallel to main.py)
# ─────────────────────────────────────────────────────────────────────────
DATA_PATH        = os.path.join(script_dir, '..', 'dataset', 'data_andre.feather')
DATE_COL         = 'date'
TARGET_COL       = 'value'

train_size       = 455
val_size         = 153
forecast_horizon = 153
lookback_window  = 30

EXOG_COLS = [
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "dom_sin", "dom_cos", "wom_sin", "wom_cos",
    "month_sin", "month_cos", "quarter_sin", "quarter_cos",
    "woy_sin", "woy_cos",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "rolling_mean_7",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_bridge_day",
]

BATCH_SIZE       = 32
EPOCHS           = 1000
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
PATIENCE         = 50
DROPOUT          = 0.1

# NHITS hyper-parameters
POOL_KERNEL_SIZES   = (8, 4, 1)    # low- -> high-frequency stacks
N_BLOCKS_PER_STACK  = 1
MLP_HIDDEN          = 256
N_MLP_LAYERS        = 2
seeds=[42]
#seeds = [42, 1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]

loss_type = 'MSELoss'





# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
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

    target_products = [26008, 907969, 907967, 213626]
    products = df[df['item_id'].isin(target_products)][['item_id', 'store_id']].drop_duplicates().values[:5]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results = []
    os.makedirs(os.path.join(script_dir, 'best_models'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, 'grid_search_plots'), exist_ok=True)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for item_id, store_id in products:
            print(f"\n{'='*70}\nSeed {seed} | item {item_id} | store {store_id}\n{'='*70}")

            df_p = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].copy()
            df_p = df_p.sort_values(DATE_COL).reset_index(drop=True)

            required = train_size + val_size + forecast_horizon
            if len(df_p) < required:
                print(f"  Skipping: {len(df_p)} rows < required {required}")
                continue

            test_start_idx  = len(df_p) - forecast_horizon
            val_start_idx   = test_start_idx - val_size
            train_start_idx = val_start_idx - train_size

            train_slice = slice(train_start_idx, val_start_idx)
            val_slice   = slice(val_start_idx,   test_start_idx)
            test_slice  = slice(test_start_idx,  None)

            train = df_p[TARGET_COL][train_slice].values.astype(np.float32)
            val   = df_p[TARGET_COL][val_slice].values.astype(np.float32)
            test  = df_p[TARGET_COL][test_slice].values.astype(np.float32)

            # Target scaler — fit on train
            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
            val_scaled   = scaler.transform(val.reshape(-1, 1)).flatten()
            # test stays unscaled; we only inverse-scale the forecast

            # Exog scaler — fit on train
            if EXOG_COLS:
                exog_train = df_p[EXOG_COLS][train_slice].values.astype(np.float32)
                exog_val   = df_p[EXOG_COLS][val_slice].values.astype(np.float32)
                exog_test  = df_p[EXOG_COLS][test_slice].values.astype(np.float32)
                exog_scaler = MinMaxScaler()
                exog_train_scaled = exog_scaler.fit_transform(exog_train)
                exog_val_scaled   = exog_scaler.transform(exog_val)
                exog_test_scaled  = exog_scaler.transform(exog_test)
            else:
                exog_train_scaled = np.zeros((len(train), 0), dtype=np.float32)
                exog_val_scaled   = np.zeros((len(val),   0), dtype=np.float32)
                exog_test_scaled  = np.zeros((len(test),  0), dtype=np.float32)

            n_exog = exog_train_scaled.shape[1]
            in_channels = 1 + n_exog

            # ── Training windows: built from the train segment alone ────────
            X_tr, fut_tr, y_tr = make_direct_windows(
                train_scaled, exog_train_scaled, lookback_window, forecast_horizon,
            )
            # ── Validation windows: lookback comes from end of train, target
            #     comes from val.  We stitch [tail(train) ‖ val] so we get a few
            #     sliding windows covering the validation segment.
            stitched_target = np.concatenate([train_scaled[-lookback_window:], val_scaled])
            stitched_exog   = np.concatenate(
                [exog_train_scaled[-lookback_window:], exog_val_scaled], axis=0,
            )
            X_va, fut_va, y_va = make_direct_windows(
                stitched_target, stitched_exog, lookback_window, forecast_horizon,
            )

            train_ds = TensorDataset(
                torch.from_numpy(X_tr), torch.from_numpy(fut_tr), torch.from_numpy(y_tr),
            )
            val_ds = TensorDataset(
                torch.from_numpy(X_va), torch.from_numpy(fut_va), torch.from_numpy(y_va),
            )
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

            # ── Model ───────────────────────────────────────────────────────
            model = NHITS(
                lookback=lookback_window,
                horizon=forecast_horizon,
                in_channels=in_channels,
                pool_kernel_sizes=POOL_KERNEL_SIZES,
                n_blocks_per_stack=N_BLOCKS_PER_STACK,
                mlp_hidden=MLP_HIDDEN,
                n_mlp_layers=N_MLP_LAYERS,
                dropout=DROPOUT,
                activation="relu",
                future_exog_len=forecast_horizon,
            ).to(device)

            model_dir = os.path.join(script_dir, 'best_models', f'seed_{seed}', loss_type)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'nhits_item{item_id}_store{store_id}.pth')

            start_train = time.time()
            t_losses, v_losses, best_epoch, best_val = train_nhits(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                patience=PATIENCE,
                best_model_path=model_path,
            )
            train_time = time.time() - start_train

            # ── Inference (direct: one forward pass on the last L lookback) ─
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # Lookback = last L scaled train+val target values; future exog = test exog
            full_target_scaled = np.concatenate([train_scaled, val_scaled])
            full_exog_scaled   = np.concatenate([exog_train_scaled, exog_val_scaled], axis=0)
            x_lookback = np.zeros((1, lookback_window, in_channels), dtype=np.float32)
            x_lookback[0, :, 0] = full_target_scaled[-lookback_window:]
            if n_exog:
                x_lookback[0, :, 1:] = full_exog_scaled[-lookback_window:]
            fut_test = exog_test_scaled[:forecast_horizon].reshape(1, forecast_horizon, n_exog)

            start_infer = time.time()
            with torch.no_grad():
                pred_scaled = model(
                    torch.from_numpy(x_lookback).to(device),
                    torch.from_numpy(fut_test).to(device) if n_exog else None,
                ).cpu().numpy().reshape(-1, 1)
            infer_time = time.time() - start_infer

            forecast = scaler.inverse_transform(pred_scaled).flatten()
            if len(forecast) != len(test):
                # Trim / pad to match (should be equal by construction)
                forecast = forecast[: len(test)]

            # ── Metrics ─────────────────────────────────────────────────────
            rmse = float(np.sqrt(mean_squared_error(test, forecast)))
            mae  = float(mean_absolute_error(test, forecast))
            bias = float(np.mean(forecast - test))
            diff_o = test[1:]     - test[:-1]
            diff_p = forecast[1:] - forecast[:-1]
            pocid  = float(((diff_o * diff_p) > 0).sum() / len(diff_o)) if len(diff_o) else 0.0
            score  = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)

            print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  bias={bias:.4f}  "
                  f"POCID={pocid:.3f}  best_epoch={best_epoch}")

            results.append({
                'seed': seed,
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
                'best_val_loss': best_val,
            })

            # ── Plot ────────────────────────────────────────────────────────
            train_index = df_p[DATE_COL][train_slice].values
            val_index   = df_p[DATE_COL][val_slice].values
            test_index  = df_p[DATE_COL][test_slice].values

            plot_dir = os.path.join(script_dir, 'grid_search_plots', f'seed_{seed}', loss_type)
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f'nhits_item{item_id}_store{store_id}.html')

            plot_results(
                train, val, test, {'NHITS': forecast},
                train_index, val_index, test_index,
                {'NHITS': t_losses}, {'NHITS': v_losses},
                target_col=TARGET_COL,
                title=f'NHITS Direct Forecast — Item {item_id} Store {store_id} (Seed {seed})',
                save_path=plot_path,
                rmse={'NHITS': rmse}, mae={'NHITS': mae}, bias={'NHITS': bias},
                score={'NHITS': score}, pocid={'NHITS': pocid}, df_full=df_p,
            )

    results_df = pd.DataFrame(results)
    out_csv = os.path.join(script_dir, 'nhits_results.csv')
    results_df.to_csv(out_csv, index=False)
    print(f"\nDone. Results saved to {out_csv}")


if __name__ == "__main__":
    main()
