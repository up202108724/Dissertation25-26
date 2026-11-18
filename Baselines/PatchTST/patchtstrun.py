"""
PatchTST trained and evaluated with its own protocol (Nie et al., ICLR 2023,
``yuqinie98/PatchTST`` ``PatchTST_supervised/exp/exp_main.py``).

Protocol
--------
* Direct multi-step forecasting: one forward pass maps the last ``SEQ_LEN``
  days to all ``forecast_horizon`` days.  No recursion, no fed-back predictions.
* Univariate (``features='S'``): the target series only, no exogenous features.
  RevIN inside the model handles per-window normalisation.
* ``StandardScaler`` fitted on the train split only.
* Windows ``(seq_len -> pred_len)`` sliding by one day.  As in the original data
  loaders, the input of a validation window may reach back into the train split;
  its targets are always validation days.
* Adam + MSE, ``OneCycleLR`` (stepped every batch, ``pct_start=0.3``), early
  stopping on the validation loss; the best-validation checkpoint is restored.
* Test: the last ``SEQ_LEN`` days before the test period -> one 152-day forecast.

Deviation: training windows are not shuffled (time-series ordering is kept).

Run from anywhere:  python Baselines/PatchTST/patchtstrun.py
"""

import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(script_dir, '..', '..'))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from patchtst import PatchTST  # noqa: E402


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
forecast_horizon = 152   # pred_len
SEQ_LEN = 30             # seq_len (lookback)

# PatchTST.  The paper uses patch 16 / stride 8 on a 336-day lookback (~41 patches,
# a patch is ~5% of the window).  Applied to a 30-day lookback that would give a
# patch of over half the window and only 3 patches, so attention would have
# almost nothing to mix.  Patch 8 / stride 4 keeps 7 patches (50% overlap, as in
# the paper), and matches the GNN PatchTST runs and the PatchTST defaults.
PATCH_LEN = 8
STRIDE = 4
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
D_FF = 64
DROPOUT = 0.2
REVIN = True

# Training hyper-parameters (paper defaults).
batch_size = 32
EPOCHS = 100             # train_epochs
PATIENCE = 10
LEARNING_RATE = 0.0001   # OneCycleLR max lr
PCT_START = 0.3

seeds = [42, 1000, 26008, 907969, 1268319, 2185791, 56918379, 1369308036]
target_products = [26008, 907969, 907967, 213626]
SAVE_PLOTS = True

MODEL_NAME = 'PatchTST'
RESULTS_CSV = os.path.join(script_dir, 'patchtst_results.csv')


def reseed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(test, forecast):
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    bias = np.mean(forecast - test)
    d_true, d_pred = np.diff(test), np.diff(forecast)
    pocid = float(((d_true * d_pred) > 0).mean()) if len(d_true) else 0.0
    score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)
    return rmse, mae, bias, score, pocid


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
class DirectWindowDataset(Dataset):
    """``series[i : i+seq_len] -> series[i+seq_len : i+seq_len+pred_len]``.

    ``first_target`` drops windows whose first target day lies before that
    index, so validation windows may read train inputs but predict only
    validation days.
    """

    def __init__(self, series, seq_len, pred_len, first_target=0):
        self.series = np.asarray(series, dtype=np.float32)
        self.seq_len, self.pred_len = seq_len, pred_len
        start = max(0, first_target - seq_len)
        self.starts = list(range(start, len(self.series) - seq_len - pred_len + 1))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]
        x = self.series[i:i + self.seq_len, None]                                  # (L, 1)
        y = self.series[i + self.seq_len:i + self.seq_len + self.pred_len, None]   # (H, 1)
        return torch.from_numpy(x), torch.from_numpy(y)


# -----------------------------------------------------------------------------
# Training (exp_main.Exp_Main.train / vali)
# -----------------------------------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += criterion(model(x), y).item() * len(x)
            n += len(x)
    return total / max(n, 1)


def train_patchtst(model, train_loader, val_loader, device, model_path):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, epochs=EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=PCT_START)
    best_val, best_epoch, wait = float('inf'), 0, 0
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(x)
            n += len(x)
        train_loss = total / max(n, 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{EPOCHS} | train {train_loss:.6f} | val {val_loss:.6f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val, best_epoch, wait = val_loss, epoch + 1, 0
            torch.save(model.state_dict(), model_path)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, train_losses, val_losses, best_epoch, time.time() - t0


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

            scaler = StandardScaler()
            scaler.fit(train.reshape(-1, 1))
            train_val_scaled = scaler.transform(np.concatenate([train, val]).reshape(-1, 1)).ravel()

            train_ds = DirectWindowDataset(train_val_scaled[:train_size], SEQ_LEN, forecast_horizon)
            val_ds = DirectWindowDataset(train_val_scaled, SEQ_LEN, forecast_horizon, first_target=train_size)
            if len(train_ds) == 0 or len(val_ds) == 0:
                raise ValueError(f"no windows: train={len(train_ds)} val={len(val_ds)} "
                                 f"(seq_len={SEQ_LEN}, pred_len={forecast_horizon})")
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            print(f"windows: train={len(train_ds)} val={len(val_ds)}")

            reseed(seed)
            model = PatchTST(
                SEQ_LEN, horizon=forecast_horizon, patch_len=PATCH_LEN, stride=STRIDE,
                d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
                dropout=DROPOUT, revin=REVIN).to(device)

            model_dir = os.path.join(script_dir, 'best_models', f'seed_{seed}')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'patchtst_item{item_id}_store{store_id}.pth')
            model, t_losses, v_losses, best_epoch, train_time = train_patchtst(
                model, train_loader, val_loader, device, model_path)

            # Direct forecast of the whole test period from the last SEQ_LEN known days.
            model.eval()
            t0 = time.time()
            with torch.no_grad():
                x = torch.tensor(train_val_scaled[-SEQ_LEN:], dtype=torch.float32, device=device)
                pred_scaled = model(x.view(1, SEQ_LEN, 1)).cpu().numpy().reshape(-1, 1)
            infer_time = time.time() - t0
            forecast = scaler.inverse_transform(pred_scaled).ravel()

            rmse, mae, bias, score, pocid = metrics(test, forecast)
            print(f"RMSE {rmse:.4f} | MAE {mae:.4f} | bias {bias:.4f} | POCID {pocid:.3f}")

            results.append({
                'seed': seed, 'model': MODEL_NAME, 'item_id': item_id, 'store_id': store_id,
                'rmse': rmse, 'mae': mae, 'bias': bias, 'score': score, 'pocid': pocid,
                'train_time': train_time, 'inference_time': infer_time, 'best_epoch': best_epoch,
            })
            # Save after every model so an interrupted run keeps its results.
            pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)

            if plot_results is not None:
                plot_dir = os.path.join(script_dir, 'plots', f'seed_{seed}')
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
