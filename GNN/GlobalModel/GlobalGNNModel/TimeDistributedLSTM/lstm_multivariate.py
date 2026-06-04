import os
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------------------------------------------------------
# Imports from sibling / parent modules
# -----------------------------------------------------------------------------
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GLOBAL_MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))  # GNN/GlobalModel
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def _load_compute_similarities():
    """Load compute_similarities_allvsall from GlobalModel/utils.py by explicit
    path, so a stray (possibly empty) utils.py on sys.path cannot shadow it."""
    utils_path = os.path.join(GLOBAL_MODEL_DIR, 'utils.py')
    spec = importlib.util.spec_from_file_location('global_model_utils', utils_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_similarities_allvsall


compute_similarities_allvsall = _load_compute_similarities()
from lstm_train import train                       # reuse the baseline trainer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', '..', '..', '..', 'dataset', 'data_smooth_erratic.feather')
)
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'multivariate_results')

DATE_COL = 'date'
TARGET_COL = 'value'

# --- Graph / clustering ------------------------------------------------------
METRIC = 'spearman'             # 'pearson' | 'spearman' | 'kendall'
SIMILARITY_THRESHOLD = 0.6      # only edges with sim >= threshold are kept
MIN_CLUSTER_SIZE = 2            # a multivariate model needs at least 2 series

# --- Chronological split (mirrors the univariate baseline) -------------------
VAL_SIZE = 154
FORECAST_HORIZON = 152
LOOKBACK_WINDOW = 7

# --- Model / optimisation ----------------------------------------------------
SEED = 42
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 1000
PATIENCE = 150
TIME_DISTRIBUTED = True         # dense supervision at every step of the window

SAVE_MODELS = True              # persist the best checkpoint per cluster


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




# =============================================================================
# Clustering: re-build the 0.6 Spearman graph and take connected components
# =============================================================================
def build_similarity_clusters(df, metric=METRIC, threshold=SIMILARITY_THRESHOLD,
                              min_cluster_size=MIN_CLUSTER_SIZE):
    """Return (clusters, df_wide, cat_labels) where ``clusters`` is a list of
    item_id lists -- one per connected component of size >= min_cluster_size.

    Construction is identical to graph_global_analysis.py: an (item_id x date)
    pivot, an all-vs-all similarity matrix, thresholded into an undirected graph.
    """
    df_wide = (
        df.pivot_table(index='item_id', columns=DATE_COL, values=TARGET_COL, aggfunc='sum')
        .fillna(0)
    )
    cat_labels = (
        df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()
        if 'cat_label' in df.columns else {}
    )

    item_ids = df_wide.index.tolist()
    all_ts = df_wide.values.astype(np.float32)            # (N, T)

    print(f"Computing {metric} similarity for {len(item_ids)} x {len(item_ids)} pairs...")
    sim_matrix = compute_similarities_allvsall(all_ts, metric=metric)

    # Threshold -> undirected graph (exclude self-loops via upper triangle)
    mask = np.triu(sim_matrix >= threshold, k=1)
    rows, cols = np.where(mask)

    G = nx.Graph()
    G.add_nodes_from(item_ids)
    for i, j in zip(rows, cols):
        G.add_edge(item_ids[i], item_ids[j], weight=float(sim_matrix[i, j]))

    # Connected components -> clusters. Keep only those big enough for a
    # multivariate model; sort each by item_id for deterministic channel order.
    clusters = [
        sorted(comp)
        for comp in nx.connected_components(G)
        if len(comp) >= min_cluster_size
    ]
    clusters.sort(key=lambda c: (-len(c), c[0]))          # biggest first

    n_connected = sum(1 for _, d in G.degree() if d > 0)
    print(f"Threshold = {threshold} | metric = {metric}")
    print(f"  Connected products : {n_connected}")
    print(f"  Edges              : {G.number_of_edges()}")
    print(f"  Clusters (>= {min_cluster_size}) : {len(clusters)} "
          f"covering {sum(len(c) for c in clusters)} products")
    return clusters, df_wide, cat_labels


# =============================================================================
# Per-series metrics
# =============================================================================
def _series_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    diff_t = y_true[1:] - y_true[:-1]
    diff_p = y_pred[1:] - y_pred[:-1]
    is_pos = (diff_t * diff_p) > 0
    pocid = float(is_pos.sum() / len(is_pos)) if len(is_pos) else 0.0
    score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)
    return rmse, mae, bias, pocid, score


# =============================================================================
# Train + evaluate ONE cluster
# =============================================================================
def run_cluster(cluster_idx, item_ids, df_wide, cat_labels, device):
    """Fit one shared multivariate LSTM on a cluster and return a list of
    per-series result rows."""
    # (K, T) -> (T, K): rows are time steps, columns are the cluster's products.
    series = df_wide.loc[item_ids].values.astype(np.float32).T            # (T, K)
    T, K = series.shape

    train_size = T - VAL_SIZE - FORECAST_HORIZON
    min_train = LOOKBACK_WINDOW + 1
    if train_size < min_train:
        print(f"[SKIP] cluster {cluster_idx}: only {T} steps, need "
              f"{min_train + VAL_SIZE + FORECAST_HORIZON}.")
        return []

    test_start = T - FORECAST_HORIZON
    val_start = test_start - VAL_SIZE

    train_raw = series[:val_start]
    val_raw = series[val_start:test_start]
    test_raw = series[test_start:]

    # Per-series MinMax scaling (one scaler, scales each column independently).
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)

    train_ds = MultivariateSeqDataset(train_scaled, None, LOOKBACK_WINDOW)
    val_ds = MultivariateSeqDataset(val_scaled, None, LOOKBACK_WINDOW)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MultivariateLSTM(
        n_series=K, exog_size=0, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, dropout=DROPOUT,
        time_distributed=TIME_DISTRIBUTED,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=PATIENCE // 3
    )
    criterion = nn.MSELoss()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, f'cluster_{cluster_idx}_multivariate_lstm.pth')

    print(f"\n=== Cluster {cluster_idx} | {K} products | {T} steps "
          f"(train={train_size}, val={VAL_SIZE}, test={FORECAST_HORIZON}) ===")
    model, _, _, best_epoch, train_time = train(
        SEED, model, train_loader, val_loader, criterion, optimizer, device,
        model_path, scheduler, PATIENCE, EPOCHS,
    )

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    if not SAVE_MODELS and os.path.exists(model_path):
        os.remove(model_path)

    forecast, infer_time = multivariate_recursive_inference(
        model, LOOKBACK_WINDOW, val_scaled, None, None, scaler, device,
        FORECAST_HORIZON,
    )

    rows = []
    for k, item_id in enumerate(item_ids):
        rmse, mae, bias, pocid, score = _series_metrics(test_raw[:, k], forecast[:, k])
        rows.append({
            'cluster': cluster_idx,
            'cluster_size': K,
            'item_id': int(item_id),
            'cat_label': cat_labels.get(item_id, 'Unknown'),
            'rmse': rmse, 'mae': mae, 'bias': bias,
            'pocid': pocid, 'score': score,
            'best_epoch': best_epoch,
            'train_time': train_time,
            'inference_time': infer_time,
        })

    mean_rmse = np.mean([r['rmse'] for r in rows])
    mean_pocid = np.mean([r['pocid'] for r in rows])
    print(f"  -> cluster {cluster_idx}: mean RMSE={mean_rmse:.4f} "
          f"mean POCID={mean_pocid:.3f} (best_epoch={best_epoch}, "
          f"{train_time:.1f}s)")
    return rows


# =============================================================================
# Main
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading data from {DATA_PATH}...")

    df = pd.read_feather(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, 'item_id']).reset_index(drop=True)

    clusters, df_wide, cat_labels = build_similarity_clusters(df)
    if not clusters:
        print("No clusters of sufficient size found. Nothing to train.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    overall_start = time.time()
    for cluster_idx, item_ids in enumerate(clusters):
        try:
            all_rows.extend(run_cluster(cluster_idx, item_ids, df_wide, cat_labels, device))
        except Exception as exc:                          # keep going on failure
            import traceback
            print(f"[ERROR] cluster {cluster_idx}: {exc}")
            traceback.print_exc()

    if all_rows:
        results = pd.DataFrame(all_rows)
        per_series_path = os.path.join(RESULTS_DIR, 'multivariate_per_series_results.csv')
        results.to_csv(per_series_path, index=False)

        summary = (
            results.groupby(['cluster', 'cluster_size'])
            .agg(mean_rmse=('rmse', 'mean'), mean_mae=('mae', 'mean'),
                 mean_bias=('bias', 'mean'), mean_pocid=('pocid', 'mean'),
                 mean_score=('score', 'mean'))
            .reset_index()
        )
        summary_path = os.path.join(RESULTS_DIR, 'multivariate_cluster_summary.csv')
        summary.to_csv(summary_path, index=False)

        print(f"\nDone in {(time.time() - overall_start) / 60:.1f} min.")
        print(f"  Per-series results : {per_series_path}")
        print(f"  Cluster summary    : {summary_path}")
        print(f"  Overall mean RMSE  : {results['rmse'].mean():.4f} "
              f"over {len(results)} series in {results['cluster'].nunique()} clusters")
    else:
        print("No results produced.")


if __name__ == '__main__':
    main()
