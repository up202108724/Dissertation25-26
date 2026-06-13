import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import compute_similarities_1vsAll, compute_distances_1vsAll, neighbourhood_graph

# ── Configuration ─────────────────────────────────────────────────────────
metric     = "spearman"
THRESHOLDS = [0.634]  # chosen to yield ~10, ~20, ~30 mean nodes in the first window
COLORS     = ['#1f77b4']   # blue for the single threshold; add more if you expand THRESHOLDS
#COLORS     = ['#1f77b4', '#ff7f0e', '#2ca02c']   # blue, orange, green
window_size = 30
step        = 1
ENABLE_Z_NORMALIZATION = True

# ── Split constants ────────────────────────────────────────────────────────
val_size_global        = 30
forecast_horizon_global = 153
train_size_global       = 761 - val_size_global - forecast_horizon_global  # 455

_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "../../../dataset/data_andre_fulfilled.feather")
TOP_PATH  = os.path.join(_HERE, "../../../dataset/top_12500.feather")

# ── Load data ──────────────────────────────────────────────────────────────
top_ids = pd.read_feather(TOP_PATH)["item_id"].unique().tolist()
print(f"Loaded {len(top_ids)} top_12500 products.")

df_raw         = pd.read_feather(DATA_PATH)
cat_labels_dict = df_raw.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()
df_wide        = df_raw.pivot_table(index='item_id', columns='date',
                                    values='value', aggfunc='sum').fillna(0)

distance_metrics = ['cid', 'dtw', 'euclidean', 'manhattan', 'lorentzian',
                    'amplitude_offset', 'slope_consistency', 'phase_invariance']
is_distance = metric in distance_metrics
comp_func   = compute_distances_1vsAll if is_distance else compute_similarities_1vsAll
m_type      = "distance" if is_distance else "similarity"

L                   = len(df_wide.columns)
global_val_start_idx = L - forecast_horizon_global - val_size_global

# ── Z-normalisation (fit on train slice, applied once) ────────────────────
if ENABLE_Z_NORMALIZATION:
    print("Applying Z-score normalisation...")
    global_train_start_idx = max(0, L - forecast_horizon_global - val_size_global - train_size_global)
    train_df_wide  = df_wide.iloc[:, global_train_start_idx:global_val_start_idx]
    df_wide_scaled = df_wide.copy()
    for item_id_iter in df_wide.index:
        z_scaler = StandardScaler()
        z_scaler.fit(train_df_wide.loc[item_id_iter].values.reshape(-1, 1))
        df_wide_scaled.loc[item_id_iter] = z_scaler.transform(
            df_wide.loc[item_id_iter].values.reshape(-1, 1)
        ).flatten()
    current_df_wide = df_wide_scaled if is_distance else df_wide
else:
    current_df_wide = df_wide

# ── Output dir ────────────────────────────────────────────────────────────
_BASE_DIR = os.path.join(_HERE, "OverallDistributions")
_SAVE_DIR = os.path.join(_BASE_DIR, f"similarity_{metric}", "threshold",
                         f"W{window_size}_S{step}_combined_thresholds")
os.makedirs(_SAVE_DIR, exist_ok=True)

TOP_N = 5   # products with the most ego-graph nodes to keep per window

# ── Per-window top-N products by ego-graph node count ─────────────────────
for threshold in THRESHOLDS:
    print(f"\n=== Threshold {threshold} ===")
    nodes_per_product = []
    window_dates      = None
    used_ids          = []   # product ids aligned with nodes_per_product rows

    for k, pid in enumerate(top_ids, 1):
        if int(pid) not in current_df_wide.index:
            continue
        print(f"  [{k}/{len(top_ids)}] product {pid}...", end='\r')
        graphs, _ = neighbourhood_graph(
            product_id=int(pid),
            metric_type=m_type,
            compute_func=comp_func,
            df=current_df_wide,
            metric=metric,
            window_size=window_size,
            threshold=float(threshold),
            percentile=None,
            step_size=step,
            cat_labels=cat_labels_dict,
            plot_dir=None,
            residuals=False,
            enable_edges_within_star=True,
            enable_second_degree=False,
            train_end_idx=global_val_start_idx,
            num_plots=0,
        )
        if len(graphs) > forecast_horizon_global:
            graphs = graphs[:-forecast_horizon_global]

        nodes_per_product.append(np.array([g.number_of_nodes() for g in graphs]))
        used_ids.append(int(pid))
        if window_dates is None:
            window_dates = pd.to_datetime([g.graph['end_date'] for g in graphs])

    print()
    if not nodes_per_product:
        print(f"  [WARN] No products produced graphs for threshold {threshold} — skipping.")
        continue

    # Align all products to the same windows, then stack: rows=products, cols=windows
    min_len      = min(len(a) for a in nodes_per_product)
    nodes_matrix = np.vstack([a[:min_len] for a in nodes_per_product])  # (P, W)
    window_dates = window_dates[:min_len]
    ids_arr      = np.array(used_ids)

    # For every window, rank products by ego-graph node count (highest first)
    # and keep the top N.  Ties are broken by ascending item_id for determinism.
    keep = min(TOP_N, nodes_matrix.shape[0])
    rows = []
    for w in range(nodes_matrix.shape[1]):
        col = nodes_matrix[:, w]
        # Sort by (-n_nodes, item_id): most nodes first, stable tie-break on id.
        order = sorted(range(len(col)), key=lambda i: (-int(col[i]), int(ids_arr[i])))[:keep]
        for rank, idx in enumerate(order, 1):
            pid = int(ids_arr[idx])
            rows.append({
                'window_idx':  w,
                'window_date': window_dates[w].date(),
                'rank':        rank,
                'item_id':     pid,
                'cat_label':   cat_labels_dict.get(pid, ''),
                'n_nodes':     int(col[idx]),
            })

    top_df = pd.DataFrame(rows)
    _csv_path = os.path.join(_SAVE_DIR, f"top{TOP_N}_products_per_window_th{threshold}.csv")
    top_df.to_csv(_csv_path, index=False)

    print(f"  → {nodes_matrix.shape[0]} products × {nodes_matrix.shape[1]} windows")
    print(f"  Saved top-{TOP_N} products per window -> {_csv_path}")

    # Which products land in the top-N most often across all windows?
    appearance = top_df.groupby('item_id').size().sort_values(ascending=False)
    print(f"  Most frequent top-{TOP_N} products (item_id: # windows):")
    for pid, cnt in appearance.head(10).items():
        print(f"    {pid}: {cnt}")
