"""
main.py — Grid search over Spearman thresholds for the Static-GCN + LSTM forecaster.

Grid
----
  metric     : spearman (fixed)
  thresholds : 0.79, 0.82, 0.85, 0.88, 0.91
  seeds      : [42]

For every (product, threshold, seed) combination:
  1. Build a global static similarity graph (once per threshold).
  2. Train StaticGCNLSTMForecaster end-to-end (GCN + LSTM).
  3. Recursively forecast the test horizon.
  4. Write metrics to CSV and save a comparison HTML plot.
"""

import sys
import os
import csv
import time
import itertools
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning, message=".*incompatible dtype.*")

script_dir = os.path.dirname(os.path.abspath(__file__))

from utils import generate_exogenous_features
from train import TrainConfig, train_static_gcn_lstm
from inference import recursive_inference

# Optional plot support — import from neighbouring SimpleGNN/LSTM if available
try:
    from plots import plot_results
    _HAS_PLOTS = True
except ImportError:
    _parent = os.path.join(script_dir, '..', 'SimpleGNN', 'LSTM')
    sys.path.insert(0, os.path.abspath(_parent))
    try:
        from plots import plot_results
        _HAS_PLOTS = True
    except ImportError:
        _HAS_PLOTS = False
        print("[main] plots.py not found — HTML plots will be skipped.")


# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(script_dir, '..', '..', '..', 'dataset', 'data_andre.feather')
DATE_COL   = 'date'
TARGET_COL = 'value'
total_size       = 761
val_size         = 153
forecast_horizon = 153
train_size       = 455
lookback_window  = 30

# ── LSTM calendar features ────────────────────────────────────────────────────
EXOG_COLS_LSTM = [
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

# ── Grid ──────────────────────────────────────────────────────────────────────
METRIC               = 'spearman'
SPEARMAN_THRESHOLDS  = [0.6,0.62,0.65]
SEEDS                = [42]

PRODUCTS_TO_TEST = [
    (26008,  6269),
    (907969, 6269),
    (907967, 6269),
    (213626, 6269),
    (911753, 6269),
]

# ── Model hyper-parameters ────────────────────────────────────────────────────
BATCH_SIZE   = 32
HIDDEN_SIZES        = (32, 16)   # (gcn_hidden, gcn_out)
LSTM_HIDDEN         = 64
LSTM_LAYERS         = 1
DROPOUT             = 0.2
EPOCHS              = 1000
LR                  = 1e-3
PATIENCE            = 150
INCLUDE_2HOP        = True       # expand ego-graph with 2nd-order neighbours (Risk-2)
GCN_LAYERS          = 2          # 1 for pure 1-hop ego-graph; 2 when INCLUDE_2HOP=True
GRAPH_CONDITIONING  = 'init'     # 'init' | 'concat'  (Risk-5 mitigation)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── CSV output ────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(script_dir, 'static_gcn_lstm_spearman_grid.csv')
CSV_COLS = [
    'product_id', 'store_id', 'seed',
    'metric', 'threshold',
    'n_nodes', 'n_edges',
    'rmse', 'mae', 'bias', 'r2_score', 'pocid',
    'best_epoch', 'train_time_s',
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_metrics(actual, forecast):
    mask = ~np.isnan(forecast)
    a, f = actual[mask], np.asarray(forecast)[mask]
    if len(a) == 0:
        return None, None, None, None, None
    rmse  = float(np.sqrt(mean_squared_error(a, f)))
    mae   = float(mean_absolute_error(a, f))
    bias  = float(np.mean(f - a))
    r2    = float(r2_score(a, f))
    d_a, d_f = a[1:] - a[:-1], f[1:] - f[:-1]
    pocid = float(((d_a * d_f) > 0).sum() / max(len(d_a), 1))
    return rmse, mae, bias, r2, pocid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading {DATA_PATH} …")
    df = pd.read_feather(DATA_PATH)
    if df.index.name == DATE_COL:
        df = df.reset_index()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, 'item_id', 'store_id']).reset_index(drop=True)
    df = generate_exogenous_features(df, exog_cols=EXOG_COLS_LSTM, date_col=DATE_COL,
                                     target_col=TARGET_COL)
    full_df = df.copy()

    # df_wide for graph construction (raw values, no scaling)
    df_wide_global = (
        full_df
        .pivot_table(index='item_id', columns=DATE_COL, values=TARGET_COL, aggfunc='sum')
        .fillna(0)
    )
    df_wide_global.columns = pd.to_datetime(df_wide_global.columns).strftime('%Y-%m-%d')

    L = df_wide_global.shape[1]
    # Column index in df_wide that marks the end of train+val
    # (= before the test period starts; graph uses all observed data up to test)
    global_graph_end_col = L - forecast_horizon

    all_item_ids = list(df_wide_global.index)

    # ── Write CSV header ───────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_COLS)

    # ── Per-product loop ───────────────────────────────────────────────────────
    for product_id, store_id in PRODUCTS_TO_TEST:
        print(f"\n{'='*70}")
        print(f"PRODUCT {product_id} | STORE {store_id}")
        print(f"{'='*70}")

        df_product = (
            full_df[(full_df['item_id'] == product_id) &
                    (full_df['store_id'] == store_id)]
            .sort_values(DATE_COL).reset_index(drop=True)
        )

        T_prod          = len(df_product)
        test_start_idx  = T_prod - forecast_horizon
        val_start_idx   = test_start_idx - val_size
        train_start_idx = val_start_idx - train_size

        train_sl = slice(train_start_idx, val_start_idx)
        val_sl   = slice(val_start_idx,   test_start_idx)
        test_sl  = slice(test_start_idx,  None)

        train_raw = df_product[TARGET_COL][train_sl].values
        val_raw   = df_product[TARGET_COL][val_sl].values
        test_raw  = df_product[TARGET_COL][test_sl].values

        # Scale exog independently of target
        exog_train_raw  = df_product[EXOG_COLS_LSTM].iloc[train_sl].values.astype(np.float32)
        exog_val_raw    = df_product[EXOG_COLS_LSTM].iloc[val_sl].values.astype(np.float32)
        exog_test_raw   = df_product[EXOG_COLS_LSTM].iloc[test_sl].values.astype(np.float32)
        exog_scaler     = MinMaxScaler()
        exog_train_sc   = exog_scaler.fit_transform(exog_train_raw)
        exog_val_sc     = exog_scaler.transform(exog_val_raw)
        exog_test_sc    = exog_scaler.transform(exog_test_raw)

        # Patch scaled exog back into a per-product copy for the train function
        df_product_lstm = df_product.copy()
        exog_idx        = df_product_lstm.columns.get_indexer(EXOG_COLS_LSTM)
        df_product_lstm.iloc[train_sl, exog_idx] = exog_train_sc
        df_product_lstm.iloc[val_sl,   exog_idx] = exog_val_sc
        df_product_lstm.iloc[test_sl,  exog_idx] = exog_test_sc

        # recent_history for inference: raw target + scaled exog
        recent_history = np.column_stack([
            val_raw[-lookback_window:].reshape(-1, 1),
            exog_val_sc[-lookback_window:],
        ])

        train_index = df_product[DATE_COL].iloc[train_sl].values
        val_index   = df_product[DATE_COL].iloc[val_sl].values
        test_index  = df_product[DATE_COL].iloc[test_sl].values

        plots_dir = os.path.join(script_dir, 'static_gcn_lstm_spearman_plots')
        os.makedirs(plots_dir, exist_ok=True)

        # ── Seed loop ─────────────────────────────────────────────────────────
        for seed in SEEDS:
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            forecasts_dict    = {}
            train_losses_dict = {}
            val_losses_dict   = {}
            rmse_dict         = {}
            mae_dict          = {}
            bias_dict         = {}
            score_dict        = {}
            pocid_dict        = {}

            # ── Threshold loop ────────────────────────────────────────────────
            for threshold in SPEARMAN_THRESHOLDS:
                print(f"\n  seed={seed} | {METRIC} threshold={threshold}")

                cfg = TrainConfig(
                    lookback     = lookback_window,
                    horizon      = 1,
                    batch_size   = BATCH_SIZE,
                    train_size   = train_size,
                    val_size     = val_size,
                    lr           = LR,
                    epochs       = EPOCHS,
                    device       = str(device),
                )

                t0 = time.time()
                (model, trained_scaler,
                 t_losses, v_losses, best_epoch,
                 graph_data, item_to_node) = train_static_gcn_lstm(
                    df                   = df_product_lstm,
                    cfg                  = cfg,
                    seed                 = seed,
                    loss_type            = 'mse',
                    product_id           = f"{product_id}_{store_id}",
                    scaler               = MinMaxScaler(),
                    target_col           = TARGET_COL,
                    exog_cols            = EXOG_COLS_LSTM,
                    df_wide              = df_wide_global,
                    all_item_ids         = all_item_ids,
                    target_item_id       = product_id,
                    metric               = METRIC,
                    threshold            = threshold,
                    train_end_idx_global = global_graph_end_col,
                    hidden_sizes         = HIDDEN_SIZES,
                    lstm_hidden          = LSTM_HIDDEN,
                    lstm_layers          = LSTM_LAYERS,
                    dropout              = DROPOUT,
                    patience             = PATIENCE,
                    test_size            = forecast_horizon,
                    include_2hop         = INCLUDE_2HOP,
                    gcn_layers           = GCN_LAYERS,
                    graph_conditioning   = GRAPH_CONDITIONING,
                )
                train_time = time.time() - t0

                target_node_idx = item_to_node[product_id]
                n_nodes = graph_data.x.shape[0]
                n_edges = graph_data.edge_index.shape[1] // 2  # undirected count
                print(f"  [graph] nodes={n_nodes}  undirected edges={n_edges}  "
                      f"(target node={target_node_idx})")

                forecast = recursive_inference(
                    model            = model,
                    scaler           = trained_scaler,
                    graph_data       = graph_data,
                    target_node_idx  = target_node_idx,
                    recent_history   = recent_history,
                    future_exog      = exog_test_sc,
                    target_channel   = 0,
                    device           = str(device),
                )

                rmse, mae, bias, r2, pocid = _compute_metrics(test_raw, forecast)
                print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
                      f"POCID={pocid:.4f}  nodes={n_nodes}  edges={n_edges}  "
                      f"best_epoch={best_epoch}")

                label = f"thr={threshold}"
                forecasts_dict[label]    = forecast
                train_losses_dict[label] = t_losses
                val_losses_dict[label]   = v_losses
                rmse_dict[label]         = rmse
                mae_dict[label]          = mae
                bias_dict[label]         = bias
                score_dict[label]        = r2
                pocid_dict[label]        = pocid

                with open(CSV_PATH, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        product_id, store_id, seed,
                        METRIC, threshold,
                        n_nodes, n_edges,
                        rmse, mae, bias, r2, pocid,
                        best_epoch, round(train_time, 2),
                    ])

            # ── Comparison plot for this product × seed ────────────────────────
            if forecasts_dict and _HAS_PLOTS:
                plot_save = os.path.join(
                    plots_dir,
                    f'item_{product_id}_store_{store_id}_seed_{seed}_comparison.html',
                )
                plot_results(
                    train=train_raw, val=val_raw, test=test_raw,
                    forecast=forecasts_dict,
                    train_index=train_index, val_index=val_index, test_index=test_index,
                    train_losses=train_losses_dict, val_losses=val_losses_dict,
                    metric=METRIC, seed=seed,
                    target_col=TARGET_COL,
                    title=(f'Static GCN+LSTM | Spearman Threshold Comparison | '
                           f'Item {product_id} | Store {store_id}'),
                    save_path=plot_save,
                    rmse=rmse_dict, mae=mae_dict, bias=bias_dict,
                    score=score_dict, pocid=pocid_dict,
                )
                print(f"  Plot saved → {plot_save}")

    print(f"\nDone. Results in {CSV_PATH}")


if __name__ == '__main__':
    main()
