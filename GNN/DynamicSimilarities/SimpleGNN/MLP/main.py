"""
main.py  –  Grid search over spearman thresholds for GNN+MLP only.

Thresholds: 0.75, 0.82, 0.85, 0.88, 0.91
Models run: SimpleGNN+MLP (train_mlp_forecaster / recursive_inference)
Results saved to: gnn_mlp_spearman_grid.csv
"""

import sys
import os
import time
import csv
import itertools
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore", category=FutureWarning, message=".*incompatible dtype.*")
script_dir = os.path.dirname(os.path.abspath(__file__))

from utils import generate_exogenous_features, compute_similarities_1vsAll, neighbourhood_graph
from train import TrainConfig, train_gnn_mlp
from gnninference import recursive_inference
from plots import plot_results

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(script_dir, '..', '..', '..', '..', 'dataset', 'data_andre.feather')
DATE_COL   = 'date'
TARGET_COL = 'value'

train_size       = 455
val_size         = 153
forecast_horizon = 153
lookback_window  = 30

# ── Features ──────────────────────────────────────────────────────────────────
NODE_FEATURES = [
    'mean7', 'mean_all', 'std_all', 'zero_ratio', 'slope', 'min_v', 'max_v',
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend",
    "rolling_mean_excl_7",
    "month", "quarter",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_monday", "is_friday",
    "is_holiday", "is_thanksgiving", "is_black_friday",
    "is_christmas", "is_christmas_eve", "is_new_year_eve",
    "is_bridge_day",
]

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
SPEARMAN_THRESHOLDS = [0.75, 0.82, 0.85, 0.88,0.91]
WINDOW_SIZES        = [15]
SEEDS               = [42]

PRODUCTS_TO_TEST = [
    (26008,  6269),
    (907969, 6269),
    (907967, 6269),
    (213626, 6269),
    (911753, 6269),
]

# ── Model hyper-parameters ────────────────────────────────────────────────────
BATCH_SIZE          = 32
HIDDEN_SIZES        = (256, 128)   # MLP hidden layer sizes — input is 30×(1+31+16)=1440
DROPOUT             = 0.2
GNN_HIDDEN_CHANNELS = 32
GNN_OUT_CHANNELS    = 16
EPOCHS              = 1000
LR                  = 0.001
PATIENCE            = 150

ENABLE_EDGES         = True
ENABLE_SECOND_DEGREE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── CSV output ────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(script_dir, "gnn_mlp_spearman_grid.csv")
CSV_COLS = [
    "product_id", "store_id", "seed",
    "metric", "threshold", "window_size",
    "enable_edges", "enable_second_degree",
    "rmse", "mae", "bias", "r2_score", "pocid",
    "best_epoch", "train_time_s",
]


def _compute_metrics(actual, forecast):
    mask = ~np.isnan(forecast)
    a, f = actual[mask], np.asarray(forecast)[mask]
    if len(a) == 0:
        return None, None, None, None, None
    rmse  = np.sqrt(mean_squared_error(a, f))
    mae   = mean_absolute_error(a, f)
    bias  = float(np.mean(f - a))
    score = r2_score(a, f)
    d_a, d_f = a[1:] - a[:-1], f[1:] - f[:-1]
    pocid = float(((d_a * d_f) > 0).sum() / max(len(d_a), 1))
    return rmse, mae, bias, score, pocid


def main():
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading {DATA_PATH} …")
    df = pd.read_feather(DATA_PATH)
    if df.index.name == DATE_COL:
        df = df.reset_index()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, "item_id", "store_id"]).reset_index(drop=True)
    df = generate_exogenous_features(df, date_col=DATE_COL, exog_cols=EXOG_COLS_LSTM)
    full_df = df.copy()

    cat_labels_dict = (full_df.drop_duplicates('item_id')
                              .set_index('item_id')['cat_label']
                              .to_dict()
                       if 'cat_label' in full_df.columns else {})

    # df_wide for similarity computation (raw values, not Z-scaled)
    df_wide_global = (full_df.pivot_table(
        index='item_id', columns=DATE_COL, values=TARGET_COL, aggfunc='sum'
    ).fillna(0))
    df_wide_global.columns = pd.to_datetime(df_wide_global.columns).strftime('%Y-%m-%d')

    L = len(df_wide_global.columns)
    global_val_start_idx = L - forecast_horizon - val_size

    # ── Write CSV header ───────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_COLS)

    # ── Per-product loop ───────────────────────────────────────────────────────
    for product_id, store_id in PRODUCTS_TO_TEST:
        print(f"\n{'='*70}")
        print(f"PRODUCT {product_id} | STORE {store_id}")
        print(f"{'='*70}")

        df_product = (full_df[(full_df['item_id'] == product_id) &
                               (full_df['store_id'] == store_id)]
                      .sort_values(DATE_COL).reset_index(drop=True))

        T_prod         = len(df_product)
        test_start_idx = T_prod - forecast_horizon
        val_start_idx  = test_start_idx - val_size
        train_start_idx = val_start_idx - train_size

        train_sl = slice(train_start_idx, val_start_idx)
        val_sl   = slice(val_start_idx,   test_start_idx)
        test_sl  = slice(test_start_idx,  None)

        train_raw = df_product[TARGET_COL][train_sl].values
        val_raw   = df_product[TARGET_COL][val_sl].values
        test_raw  = df_product[TARGET_COL][test_sl].values

        # MinMax scaler for target
        scaler = MinMaxScaler()
        scaler.fit(train_raw.reshape(-1, 1))

        # LSTM exog — scaled independently
        exog_train_raw = df_product[EXOG_COLS_LSTM].iloc[train_sl].values.copy()
        exog_val_raw   = df_product[EXOG_COLS_LSTM].iloc[val_sl].values.copy()
        exog_test_raw  = df_product[EXOG_COLS_LSTM].iloc[test_sl].values.copy()

        exog_scaler = MinMaxScaler()
        exog_train_scaled = exog_scaler.fit_transform(exog_train_raw)
        exog_val_scaled   = exog_scaler.transform(exog_val_raw)
        exog_test_scaled  = exog_scaler.transform(exog_test_raw)

        # Patch scaled exog into a product copy for train functions
        df_product_lstm = df_product.copy()
        lstm_exog_idx   = df_product_lstm.columns.get_indexer(EXOG_COLS_LSTM)
        df_product_lstm.iloc[train_sl, lstm_exog_idx] = exog_train_scaled
        df_product_lstm.iloc[val_sl,   lstm_exog_idx] = exog_val_scaled
        df_product_lstm.iloc[test_sl,  lstm_exog_idx] = exog_test_scaled

        # Lookback history for inference (last lookback values of validation set)
        recent_history = np.column_stack([
            val_raw[-lookback_window:].reshape(-1, 1),
            exog_val_scaled[-lookback_window:],
        ])

        past_dates   = pd.to_datetime(df_product[DATE_COL][:test_start_idx]).dt.strftime('%Y-%m-%d').values
        future_dates = pd.to_datetime(df_product[DATE_COL][test_start_idx:]).dt.strftime('%Y-%m-%d').values

        train_index = df_product[DATE_COL].iloc[train_sl].values
        val_index   = df_product[DATE_COL].iloc[val_sl].values
        test_index  = df_product[DATE_COL].iloc[test_sl].values

        plots_dir = os.path.join(script_dir, 'gnn_mlp_spearman_plots')
        os.makedirs(plots_dir, exist_ok=True)

        # ── Seed / threshold grid ──────────────────────────────────────────────
        for seed in SEEDS:
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Collect results for comparison plot
            forecasts_dict    = {}
            train_losses_dict = {}
            val_losses_dict   = {}
            rmse_dict         = {}
            mae_dict          = {}
            bias_dict         = {}
            score_dict        = {}
            pocid_dict        = {}

            for threshold, window_sz in itertools.product(SPEARMAN_THRESHOLDS, WINDOW_SIZES):
                print(f"\n  seed={seed} | spearman threshold={threshold} | window={window_sz}")

                # Build graphs
                graphs_list, fixed_threshold = neighbourhood_graph(
                    product_id=product_id,
                    df=df_wide_global,
                    metric='spearman',
                    metric_type='similarity',
                    window_size=window_sz,
                    compute_func=compute_similarities_1vsAll,
                    threshold=threshold,
                    percentile=None,
                    step_size=1,
                    cat_labels=cat_labels_dict,
                    residuals=False,
                    enable_edges_within_star=ENABLE_EDGES,
                    enable_second_degree=ENABLE_SECOND_DEGREE,
                    train_end_idx=val_start_idx,
                )
                print(f"  resolved threshold={fixed_threshold}")

                cfg = TrainConfig(
                    lookback=lookback_window,
                    horizon=1,
                    batch_size=BATCH_SIZE,
                    train_size=train_size,
                    val_size=val_size,
                    lr=LR,
                    epochs=EPOCHS,
                    device=str(device),
                )

                t0 = time.time()
                # gnn_in_channels = graph_window_size raw ts + 8 stat features
                gnn_in_channels = window_sz + 8
                model, trained_scaler, t_losses, v_losses, best_epoch = train_gnn_mlp(
                    df=df_product_lstm,
                    cfg=cfg,
                    seed=seed,
                    loss_type='mse',
                    product_id=f"{product_id}_{store_id}",
                    scaler=MinMaxScaler(),
                    target_channel=0,
                    hidden_sizes=HIDDEN_SIZES,
                    target_col=TARGET_COL,
                    exog_cols=EXOG_COLS_LSTM,
                    graphs=graphs_list,
                    test_size=forecast_horizon,
                    gnn_in_channels=gnn_in_channels,
                    gnn_hidden_channels=GNN_HIDDEN_CHANNELS,
                    gnn_out_channels=GNN_OUT_CHANNELS,
                )
                

                forecast = recursive_inference(
                    model=model,
                    scaler=trained_scaler,
                    recent_history=recent_history,
                    future_exog=exog_test_scaled,
                    target_channel=0,
                    device=str(device),
                    df_wide=df_wide_global,
                    cat_labels=cat_labels_dict,
                    target_id=product_id,
                    metric='spearman',
                    fixed_threshold=fixed_threshold,
                    enable_edges_within_star=ENABLE_EDGES,
                    enable_second_degree=ENABLE_SECOND_DEGREE,
                    past_dates=past_dates,
                    future_dates=future_dates,
                    graph_window_size=window_sz,
                )

                rmse, mae, bias, r2, pocid = _compute_metrics(test_raw, forecast)
                print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
                      f"POCID={pocid:.4f}  best_epoch={best_epoch}")

                label = f"spearman_{threshold}"
                forecasts_dict[label]    = forecast
                train_losses_dict[label] = t_losses
                val_losses_dict[label]   = v_losses
                rmse_dict[label]         = rmse
                mae_dict[label]          = mae
                bias_dict[label]         = bias
                score_dict[label]        = r2
                pocid_dict[label]        = pocid

            # ── Comparison plot for this product × seed ────────────────────────
            if forecasts_dict:
                plot_save = os.path.join(
                    plots_dir,
                    f'item_{product_id}_store_{store_id}_seed_{seed}_comparison.html',
                )
                plot_results(
                    train=train_raw, val=val_raw, test=test_raw,
                    forecast=forecasts_dict,
                    train_index=train_index, val_index=val_index, test_index=test_index,
                    train_losses=train_losses_dict, val_losses=val_losses_dict,
                    metric='spearman', seed=seed,
                    target_col=TARGET_COL,
                    title=f'GCN+MLP Spearman Threshold Comparison | Item {product_id} | Store {store_id}',
                    save_path=plot_save,
                    rmse=rmse_dict, mae=mae_dict, bias=bias_dict,
                    score=score_dict, pocid=pocid_dict,
                )
                print(f"  Plot saved → {plot_save}")

    print(f"\nDone. Results saved to {CSV_PATH}")


if __name__ == '__main__':
    main()
