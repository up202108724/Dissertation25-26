"""
Data preparation and result bookkeeping shared by the Graph2Vec experiment runners
(LSTM/graph2vec_lstm.py, LSTM/graph2vec_allproducts.py and the TimeDistributedMLP runners).
"""
import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Graph2vec.graph_utils import DISTANCE_METRICS

RESULT_FIELDS = ('forecasts', 'train_losses', 'val_losses', 'rmse', 'mae', 'bias', 'score', 'pocid')


# ── Data preparation ──────────────────────────────────────────────────────────
def load_panel(data_path, exog_cols, generate_exogenous_features, date_col='date', target_col='value'):
    """
    Load the long-format sales panel, add exogenous features and pivot it to the wide
    (item_id x date) matrix used for graph construction.

    Returns (full_df, cat_labels, df_wide).
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_feather(data_path)
    if date_col in df.index.names:
        df = df.reset_index(drop=True) if date_col in df.columns else df.reset_index()
    df = df.reset_index(drop=True)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col, 'item_id', 'store_id']).reset_index(drop=True)
    df = generate_exogenous_features(df, date_col=date_col, exog_cols=exog_cols)

    cat_labels = (
        df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()
        if 'cat_label' in df.columns else {}
    )
    df_wide = df.pivot_table(index='item_id', columns=date_col, values=target_col, aggfunc='sum').fillna(0)
    df_wide.columns = pd.to_datetime(df_wide.columns).strftime('%Y-%m-%d')
    return df, cat_labels, df_wide


def build_product_scalers(df_wide, train_size, val_size, forecast_horizon):
    """
    Fit one StandardScaler per item on the global training window and z-score the whole
    wide matrix with it (used by distance metrics).

    Returns (product_scalers, df_wide_scaled).
    """
    L = len(df_wide.columns)
    train_start = max(0, L - forecast_horizon - val_size - train_size)
    val_start = L - forecast_horizon - val_size
    train_wide = df_wide.iloc[:, train_start:val_start]

    product_scalers = {}
    df_wide_scaled = df_wide.copy()
    for item_id in df_wide.index:
        z_scaler = StandardScaler()
        z_scaler.fit(train_wide.loc[item_id].values.reshape(-1, 1))
        product_scalers[item_id] = z_scaler
        df_wide_scaled.loc[item_id] = z_scaler.transform(
            df_wide.loc[item_id].values.reshape(-1, 1)
        ).flatten()
    return product_scalers, df_wide_scaled


def select_df_wide(metric, df_wide, df_wide_scaled):
    """Distance metrics are computed on the z-scored matrix, similarity metrics on raw values."""
    return df_wide_scaled if metric in DISTANCE_METRICS else df_wide


def split_and_scale(df_p, target_col, exog_cols, train_size, val_size, forecast_horizon):
    """
    Split one product's series into the last `train_size | val_size | forecast_horizon`
    rows and MinMax-scale target and exogenous columns on the training split.
    """
    test_start_idx = len(df_p) - forecast_horizon
    val_start_idx = test_start_idx - val_size
    train_start_idx = val_start_idx - train_size

    train_slice = slice(train_start_idx, val_start_idx)
    val_slice = slice(val_start_idx, test_start_idx)
    test_slice = slice(test_start_idx, None)

    train = df_p[target_col][train_slice].values
    val = df_p[target_col][val_slice].values
    test = df_p[target_col][test_slice].values

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test.reshape(-1, 1)).flatten()

    if exog_cols:
        exog_scaler = MinMaxScaler()
        exog_train_scaled = exog_scaler.fit_transform(df_p[exog_cols][train_slice].values)
        exog_val_scaled = exog_scaler.transform(df_p[exog_cols][val_slice].values)
        exog_test_scaled = exog_scaler.transform(df_p[exog_cols][test_slice].values)
    else:
        exog_scaler = exog_train_scaled = exog_val_scaled = exog_test_scaled = None

    return SimpleNamespace(
        test_start_idx=test_start_idx, val_start_idx=val_start_idx, train_start_idx=train_start_idx,
        train_slice=train_slice, val_slice=val_slice, test_slice=test_slice,
        train=train, val=val, test=test,
        scaler=scaler, train_scaled=train_scaled, val_scaled=val_scaled, test_scaled=test_scaled,
        exog_scaler=exog_scaler, exog_train_scaled=exog_train_scaled,
        exog_val_scaled=exog_val_scaled, exog_test_scaled=exog_test_scaled,
    )


# ── Metrics & result bookkeeping ──────────────────────────────────────────────
def forecast_metrics(y_true, y_pred):
    """Returns (rmse, mae, bias, r2, pocid); all None when there is nothing to score."""
    if len(y_true) == 0:
        return None, None, None, None, None
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    r2 = r2_score(y_true, y_pred)
    is_positive = ((y_true[1:] - y_true[:-1]) * (y_pred[1:] - y_pred[:-1])) > 0
    pocid = is_positive.sum() / len(is_positive) if len(is_positive) > 0 else 0.0
    return rmse, mae, bias, r2, pocid


def new_result_bucket():
    """One {label: value} dict per result field, plus the resolved graph threshold."""
    bucket = {field: {} for field in RESULT_FIELDS}
    bucket['threshold'] = None
    return bucket


def record_result(bucket, label, values):
    """Store one run's values (a dict keyed by RESULT_FIELDS) under `label`."""
    for field in RESULT_FIELDS:
        bucket[field][label] = values.get(field)


def merge_result_buckets(target, source):
    for field in RESULT_FIELDS:
        target.setdefault(field, {}).update(source[field])


def config_label(is_threshold_mode, threshold, percentile, resolved_threshold,
                 window_size, step_size, enable_edges, enable_second_degree):
    """Legend label for one graph configuration."""
    if is_threshold_mode:
        param_str = f"th:{threshold}"
    else:
        th_str = f"{resolved_threshold:.4f}" if resolved_threshold is not None else "N/A"
        param_str = f"pct:{percentile} (val:{th_str})"
    return f"{param_str}|w:{window_size}|st:{step_size}|e:{enable_edges}|2nd:{enable_second_degree}"


def param_grid_hash(thresholds, percentiles):
    """Short, stable folder name for a threshold / percentile grid."""
    if thresholds is not None and len(thresholds) > 0 and percentiles is None:
        raw_str = "_".join(map(str, thresholds))
    else:
        raw_str = "_".join(map(str, percentiles))
    return hashlib.md5(raw_str.encode()).hexdigest()[:8]


def append_result_row(csv_path, row):
    """
    Append one result dict to `csv_path`. If the file already exists its header is kept
    (columns missing from `row` are left empty), so older results files stay aligned.
    """
    import csv

    fieldnames = list(row)
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            fieldnames = next(csv.reader(f), None) or fieldnames
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', restval='')
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Reporting ─────────────────────────────────────────────────────────────────
def plot_threshold_correlations(results_dir, skip=('no_emb.csv',)):
    """For every results CSV in `results_dir`, plot threshold (or percentile) vs RMSE and MAE."""
    import matplotlib.pyplot as plt

    print("\nGenerating Correlation Plots across all collected CSVs...")
    for csv_file in os.listdir(results_dir):
        if not csv_file.endswith('.csv') or csv_file in skip:
            continue
        metric_name = csv_file.replace('.csv', '')
        try:
            res_df = pd.read_csv(os.path.join(results_dir, csv_file))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'Threshold vs RMSE and MAE | Metric: {metric_name}', fontsize=16)

            x_col = 'threshold' if res_df['threshold'].notna().any() else 'percentile'
            plot_data = res_df.dropna(subset=[x_col, 'rmse', 'mae']).sort_values(by=x_col)
            if plot_data.empty:
                continue

            ax1.plot(plot_data[x_col], plot_data['rmse'], marker='o', linestyle='-', color='b')
            ax1.set_title(f'{x_col.capitalize()} vs RMSE')
            ax1.set_xlabel(x_col.capitalize())
            ax1.set_ylabel('RMSE')
            ax1.grid(True)

            ax2.plot(plot_data[x_col], plot_data['mae'], marker='s', linestyle='-', color='r')
            ax2.set_title(f'{x_col.capitalize()} vs MAE')
            ax2.set_xlabel(x_col.capitalize())
            ax2.set_ylabel('MAE')
            ax2.grid(True)

            plot_save_path = os.path.join(results_dir, f"{metric_name}_correlation_plot.png")
            plt.tight_layout()
            plt.savefig(plot_save_path)
            plt.close()
            print(f"Saved correlation plot for {metric_name} at {plot_save_path}")
        except Exception as e:
            print(f"Failed to generate plot for {csv_file}: {e}")
