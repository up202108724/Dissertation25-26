import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, '..')))

from utils import compute_similarities_1vsAll, compute_distances_1vsAll, neighbourhood_graph
from graph_plot import plot_networkx_plotly

# ── Configuration ─────────────────────────────────────────────────────────
metric      = "spearman"
threshold   = 0.8
window_size = 30
step        = 1

PRODUCTS_TO_TEST = [(907969,6269)]  # How many products to generate graphs for (evenly sampled from top_ids)
# How many graphs to draw per product (evenly sampled across training windows)
NUM_GRAPHS_PER_PRODUCT = 10

ENABLE_Z_NORMALIZATION = True

# ── Split constants ────────────────────────────────────────────────────────
val_size_global         = 30
forecast_horizon_global = 153
train_size_global       = 761 - val_size_global - forecast_horizon_global  # 578

DATA_PATH = os.path.join(_HERE, "../../../dataset/data_andre_fulfilled.feather")
TOP_PATH  = os.path.join(_HERE, "../../../dataset/top_12500.feather")

# ── Load data ──────────────────────────────────────────────────────────────
top_ids = pd.read_feather(TOP_PATH)["item_id"].unique().tolist()
print(f"Loaded {len(top_ids)} products.")

df_raw          = pd.read_feather(DATA_PATH)
cat_labels_dict = df_raw.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()
df_wide         = df_raw.pivot_table(index='item_id', columns='date',
                                     values='value', aggfunc='sum').fillna(0)

distance_metrics = ['cid', 'dtw', 'euclidean', 'manhattan', 'lorentzian',
                    'amplitude_offset', 'slope_consistency', 'phase_invariance']
is_distance = metric in distance_metrics
comp_func   = compute_distances_1vsAll if is_distance else compute_similarities_1vsAll
m_type      = "distance" if is_distance else "similarity"

L                    = len(df_wide.columns)
global_val_start_idx = L - forecast_horizon_global - val_size_global

# ── Z-normalisation (fit on train slice) ─────────────────────────────────
if ENABLE_Z_NORMALIZATION:
    print("Applying Z-score normalisation...")
    global_train_start_idx = max(0, global_val_start_idx - train_size_global)
    train_df_wide  = df_wide.iloc[:, global_train_start_idx:global_val_start_idx]
    df_wide_scaled = df_wide.copy()
    for iid in df_wide.index:
        scaler = StandardScaler()
        scaler.fit(train_df_wide.loc[iid].values.reshape(-1, 1))
        df_wide_scaled.loc[iid] = scaler.transform(
            df_wide.loc[iid].values.reshape(-1, 1)
        ).flatten()
    current_df_wide = df_wide_scaled if is_distance else df_wide
else:
    current_df_wide = df_wide

# ── Output dir ────────────────────────────────────────────────────────────
SAVE_DIR = os.path.join(_HERE, "DynamicGraphPlots",
                        f"{metric}_th{threshold}_W{window_size}_S{step}")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Per-product graph generation & plotting ───────────────────────────────
for k, (pid, *_) in enumerate(PRODUCTS_TO_TEST, 1):
    pid = int(pid)
    if pid not in current_df_wide.index:
        print(f"[{k}/{len(PRODUCTS_TO_TEST)}] product {pid} not in dataframe — skipping.")
        continue
    print(f"[{k}/{len(PRODUCTS_TO_TEST)}] product {pid}")

    graphs, _ = neighbourhood_graph(
        product_id=pid,
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

    # Keep only training windows (strip forecast horizon tail)
    if len(graphs) > forecast_horizon_global:
        graphs = graphs[:-forecast_horizon_global]

    if not graphs:
        print(f"  [WARN] No graphs produced — skipping.")
        continue

    # Take the first NUM_GRAPHS_PER_PRODUCT windows chronologically
    graphs_to_plot = graphs[:NUM_GRAPHS_PER_PRODUCT]

    prod_dir = os.path.join(SAVE_DIR, str(pid))
    os.makedirs(prod_dir, exist_ok=True)

    for idx, G in enumerate(graphs_to_plot):
        end_date = G.graph.get('end_date', None)
        date_str = pd.Timestamp(end_date).strftime('%Y-%m-%d') if end_date else f'w{idx:03d}'

        title  = (f"Product {pid} | {metric} τ={threshold} | "
                  f"window ending {date_str} | "
                  f"nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
        fname  = os.path.join(prod_dir, f"{pid}_{date_str}.html")

        plot_networkx_plotly(G, title=title, save_path=fname, target_node=pid)
        print(f"  saved {fname}")

print(f"\nDone. Plots saved under {SAVE_DIR}")