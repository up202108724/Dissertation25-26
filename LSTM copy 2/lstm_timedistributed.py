"""
Time-distributed (many-to-many) LSTM head vs. the standard seq-to-one head.

This is a self-contained experiment so it does NOT touch the shared lstm.py /
dataset.py / LSTM.lstm_train / LSTM.lstm_inference modules that the other
scripts rely on. It only reuses plot_results and generate_exogenous_features.

Two heads are trained and compared per product:

  * 'seq2one'        : last hidden state -> Linear -> 1   (the current model)
  * 'timedistributed': Linear applied to EVERY step -> (B, seq, 1)

The time-distributed head is supervised at every step: for an input window
[t-seq+1 .. t] it predicts [t-seq+2 .. t+1], i.e. each position forecasts the
next value. This gives one loss term per step instead of one per window (denser
supervision). For forecasting we still read only the last step's output, so
recursive multi-step inference is identical to the seq-to-one case.
"""

import sys
import os
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm import LSTMFlexible
from dataset import SeqTargetDataset
sys.path.append(os.path.abspath('..'))
from plots import plot_results
from utils import generate_exogenous_features

# -----------------------------------------------------------------------------
# Configuration (mirrors lstm_strategies_loss.py)
# -----------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, '..', 'dataset', 'data_andre.feather')
DATE_COL = 'date'
TARGET_COL = 'value'


val_size = 31
forecast_horizon = 153
train_size = 761 - forecast_horizon - val_size
lookback_window = 30

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

batch_size = 32
hidden_size = 32
num_layers = 1
dropout = 0.0
EPOCHS = 1000
LEARNING_RATE = 0.001
PATIENCE = 150
seeds = [42]

# Which heads to run/compare
HEADS = ['lstm_simple','seq2one', 'timedistributed']





# -----------------------------------------------------------------------------
# Training: dense supervision for the time-distributed head, last-step only for
# seq-to-one. Validation is ALWAYS monitored on the last-step MSE so the two
# heads are directly comparable and early stopping is consistent.
# -----------------------------------------------------------------------------
def train(seed, model, train_loader, val_loader, criterion, optimizer, device,
          best_model_path, scheduler=None, patience=10, epochs=1000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    val_criterion = nn.MSELoss()
    td = model.time_distributed

    start = time.time()
    train_losses, val_losses = [], []
    best_val = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:                 # by: (B, seq, 1)
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            if td:
                loss = criterion(out, by)           # all steps
            else:
                loss = criterion(out, by[:, -1, :])  # last step only
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # Validation: last-step prediction, MSE, for both heads.
        model.eval()
        outs, tgts = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                last = out[:, -1, :] if td else out
                outs.append(last)
                tgts.append(by[:, -1, :])
        outs = torch.cat(outs, dim=0).view(-1)
        tgts = torch.cat(tgts, dim=0).view(-1)
        val_loss = val_criterion(outs, tgts).item()
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Train {avg_train:.6f} | Val {val_loss:.6f}")

    return model, train_losses, val_losses, best_epoch, time.time() - start


# -----------------------------------------------------------------------------
# Recursive inference. Reads the LAST timestep's output for either head, so it
# is identical to the seq-to-one pipeline. Dynamic lag features are recomputed
# from the running prediction history, mirroring LSTM.lstm_inference.
# -----------------------------------------------------------------------------
def recursive_inference(model, seq_length, val_scaled, exog_val_scaled,
                        exog_test_scaled, exog_test, scaler, exog_scaler,
                        device, exog_cols, forecast_window):
    model.eval()
    td = model.time_distributed

    current_seq = val_scaled[-seq_length:].tolist()
    current_exog_seq = []
    if exog_cols:
        current_exog_seq = exog_val_scaled[-seq_length + 1:].tolist() + [exog_test_scaled[0].tolist()]

    dynamic_features = []
    if exog_cols:
        for idx, col in enumerate(exog_cols):
            try:
                if col.startswith("rolling_mean_excl_"):
                    dynamic_features.append((idx, "rolling_mean_excl", int(col.split("_")[-1])))
                elif col.startswith("rolling_mean_"):
                    dynamic_features.append((idx, "rolling_mean", int(col.split("_")[-1])))
                elif col.startswith("lag_"):
                    dynamic_features.append((idx, "lag", int(col.split("_")[-1])))
            except ValueError:
                pass

    forecast = []
    start = time.time()
    with torch.no_grad():
        for step in range(forecast_window):
            if exog_cols:
                x_np = np.column_stack([np.array(current_seq).reshape(-1, 1),
                                        np.array(current_exog_seq)])
            else:
                x_np = np.array(current_seq).reshape(-1, 1)

            x = torch.FloatTensor(x_np).unsqueeze(0).to(device)
            out = model(x)
            pred = (out[0, -1, 0] if td else out[0, 0]).cpu().item()
            forecast.append(pred)

            current_seq = current_seq[1:] + [pred]

            if exog_cols and step + 1 < forecast_window:
                next_exog_raw = exog_test[step + 1].copy()
                if dynamic_features:
                    max_w = max(w for _, _, w in dynamic_features)
                    hist = scaler.inverse_transform(
                        np.array(current_seq[-max_w:]).reshape(-1, 1)).flatten()
                    for idx, feat_type, w in dynamic_features:
                        if feat_type == "lag":
                            next_exog_raw[idx] = hist[-w] if w <= len(hist) else (hist[0] if len(hist) else 0.0)
                        else:  # rolling_mean / rolling_mean_excl
                            win = hist[-w:]
                            next_exog_raw[idx] = np.mean(win) if len(win) else 0.0
                next_exog_scaled = exog_scaler.transform(next_exog_raw.reshape(1, -1))[0]
                current_exog_seq = current_exog_seq[1:] + [next_exog_scaled.tolist()]

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    return forecast, time.time() - start


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
    target_products = [26008,911753]
    #target_products = [26008, 907969, 907967, 213626]
    products = df[df['item_id'].isin(target_products)][['item_id', 'store_id']].drop_duplicates().values[:5]

    os.makedirs('best_models', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results = []
    input_size = 1 + len(EXOG_COLS)

    for seed in seeds:
        for item_id, store_id in products:
            df_product = df[(df['item_id'] == item_id) & (df['store_id'] == store_id)].copy()
            df_product = df_product.sort_values(DATE_COL).reset_index(drop=True)

            test_start_idx = len(df_product) - forecast_horizon
            val_start_idx = test_start_idx - val_size
            train_start_idx = val_start_idx - train_size

            train_slice = slice(train_start_idx, val_start_idx)
            val_slice = slice(val_start_idx, test_start_idx)
            test_slice = slice(test_start_idx, None)

            train_v = df_product[TARGET_COL][train_slice].values
            val_v = df_product[TARGET_COL][val_slice].values
            test_v = df_product[TARGET_COL][test_slice].values

            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_v.reshape(-1, 1)).flatten()
            val_scaled = scaler.transform(val_v.reshape(-1, 1)).flatten()

            exog_scaler = MinMaxScaler()
            exog_train_scaled = exog_scaler.fit_transform(df_product[EXOG_COLS][train_slice].values)
            exog_val_scaled = exog_scaler.transform(df_product[EXOG_COLS][val_slice].values)
            exog_test_scaled = exog_scaler.transform(df_product[EXOG_COLS][test_slice].values)
            exog_test = df_product[EXOG_COLS][test_slice].values

            train_ds = SeqTargetDataset(train_scaled, exog_train_scaled, lookback_window)
            val_ds = SeqTargetDataset(val_scaled, exog_val_scaled, lookback_window)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

            head_forecasts, head_tl, head_vl = {}, {}, {}
            head_rmse, head_mae, head_bias, head_score, head_pocid = {}, {}, {}, {}, {}

            train_index = df_product[DATE_COL][train_slice].values
            val_index = df_product[DATE_COL][val_slice].values
            test_index = df_product[DATE_COL][test_slice].values

            for head in HEADS:
                print(f"\n--- Head: {head}, Seed: {seed}, Item: {item_id}, Store: {store_id} ---")
                model = LSTMFlexible(input_size=input_size, hidden_size=hidden_size,
                                     num_layers=num_layers, dropout=dropout,
                                     time_distributed=(head == 'timedistributed')).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=PATIENCE // 3)

                model_dir = os.path.join(script_dir, 'best_models', f'seed_{seed}', 'timedistributed_cmp', head)
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f'lstm_item{item_id}_store{store_id}.pth')

                model, t_losses, v_losses, best_epoch, train_time = train(
                    seed, model, train_loader, val_loader, nn.MSELoss(), optimizer,
                    device, model_path, scheduler, PATIENCE, EPOCHS)

                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path))

                forecast, infer_time = recursive_inference(
                    model, lookback_window, val_scaled, exog_val_scaled,
                    exog_test_scaled, exog_test, scaler, exog_scaler,
                    device, EXOG_COLS, forecast_horizon)

                rmse = float(np.sqrt(mean_squared_error(test_v, forecast)))
                mae = float(mean_absolute_error(test_v, forecast))
                bias = float(np.mean(forecast - test_v))
                d_o = test_v[1:] - test_v[:-1]
                d_p = forecast[1:] - forecast[:-1]
                is_pos = (d_o * d_p) > 0
                pocid = float(is_pos.sum() / len(is_pos)) if len(is_pos) else 0.0
                score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)

                head_forecasts[head] = forecast
                head_tl[head] = t_losses
                head_vl[head] = v_losses
                head_rmse[head] = rmse
                head_mae[head] = mae
                head_bias[head] = bias
                head_score[head] = score
                head_pocid[head] = pocid

                results.append({
                    'seed': seed, 'head': head, 'item_id': item_id, 'store_id': store_id,
                    'rmse': rmse, 'mae': mae, 'bias': bias, 'pocid': pocid, 'score': score,
                    'train_time': train_time, 'inference_time': infer_time, 'best_epoch': best_epoch,
                })
                print(f"   -> rmse={rmse:.4f} mae={mae:.4f} bias={bias:.4f} pocid={pocid:.3f}")

            plot_dir = os.path.join(script_dir, 'grid_search_plots', f'seed_{seed}', 'timedistributed_cmp')
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f'item{item_id}_store{store_id}_head_comparison.html')
            plot_results(train_v, val_v, test_v, head_forecasts,
                         train_index, val_index, test_index, head_tl, head_vl,
                         target_col=TARGET_COL,
                         title=f'LSTM Head Comparison - Item {item_id} Store {store_id} (Seed {seed})',
                         save_path=plot_path,
                         rmse=head_rmse, mae=head_mae, bias=head_bias,
                         score=head_score, pocid=head_pocid, df_full=df_product)

    pd.DataFrame(results).to_csv(
        os.path.join(script_dir, 'lstm_timedistributed_results.csv'), index=False)
    print("\nDone. Results saved to lstm_timedistributed_results.csv.")


if __name__ == "__main__":
    main()
