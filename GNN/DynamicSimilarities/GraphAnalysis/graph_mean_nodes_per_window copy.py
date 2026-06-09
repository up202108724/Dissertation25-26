import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath('..'))
from utils import compute_similarities_1vsAll, compute_distances_1vsAll, neighbourhood_graph

# ── Configuration (must match graph construction settings) ────────────────
metric = "spearman"          # distance/similarity metric
window_size = 30
step = 1
threshold = "0.60"
ENABLE_Z_NORMALIZATION = True

# ── Split constants — must match main_gcn_mlp.py exactly ──────────────────
val_size_global = 30
forecast_horizon_global = 153
train_size_global = 761 - val_size_global - forecast_horizon_global  # 455

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "../../../dataset/data_andre_fulfilled.feather")
TOP_PATH = os.path.join(_HERE, "../../../dataset/top_12500.feather")

# ── Load the 61 benchmark product ids ─────────────────────────────────────
top_ids = pd.read_feather(TOP_PATH)["item_id"].unique().tolist()
print(f"Loaded {len(top_ids)} top_12500 products.")

# ── Build the wide matrix once (shared windowing/dates across all products) ─
df_raw = pd.read_feather(DATA_PATH)
cat_labels_dict = df_raw.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()
df_wide = df_raw.pivot_table(index='item_id', columns='date', values='value', aggfunc='sum').fillna(0)

distance_metrics = ['cid', 'dtw', 'euclidean', 'manhattan', 'lorentzian',
                    'amplitude_offset', 'slope_consistency', 'phase_invariance']
is_distance = metric in distance_metrics
comp_func = compute_distances_1vsAll if is_distance else compute_similarities_1vsAll
m_type = "distance" if is_distance else "similarity"

L = len(df_wide.columns)
global_val_start_idx = L - forecast_horizon_global - val_size_global

if ENABLE_Z_NORMALIZATION:
    print("Applying Z-score normalization (fit on train portion only)...")
    global_train_start_idx = max(0, L - forecast_horizon_global - val_size_global - train_size_global)
    train_df_wide = df_wide.iloc[:, global_train_start_idx:global_val_start_idx]

    df_wide_scaled = df_wide.copy()
    for item_id_iter in df_wide.index:
        z_scaler = StandardScaler()
        z_scaler.fit(train_df_wide.loc[item_id_iter].values.reshape(-1, 1))
        full_ts = df_wide.loc[item_id_iter].values.reshape(-1, 1)
        df_wide_scaled.loc[item_id_iter] = z_scaler.transform(full_ts).flatten()
    current_df_wide = df_wide_scaled if is_distance else df_wide
else:
    print("Z-score normalization disabled. Using raw values.")
    current_df_wide = df_wide

# ── Compute nodes-per-window for every product ────────────────────────────
nodes_per_product = []   # list of np.array (one per product), one value per window
window_dates = None      # reference dates (end_date of each window), shared across products
used_ids = []

for k, pid in enumerate(top_ids, 1):
    if int(pid) not in current_df_wide.index:
        print(f"  [{k}/{len(top_ids)}] product {pid} not in matrix — skipping.")
        continue
    print(f"  [{k}/{len(top_ids)}] building graphs for product {pid}...")
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
    # Drop the test period so we only study Train + Validation windows
    if len(graphs) > forecast_horizon_global:
        graphs = graphs[:-forecast_horizon_global]

    nodes_per_product.append(np.array([g.number_of_nodes() for g in graphs]))
    if window_dates is None:
        window_dates = pd.to_datetime([g.graph['end_date'] for g in graphs])
    used_ids.append(pid)

# Align all products to the same number of windows (defensive trim) and stack
min_len = min(len(a) for a in nodes_per_product)
nodes_matrix = np.vstack([a[:min_len] for a in nodes_per_product])  # [n_products x n_windows]
window_dates = window_dates[:min_len]

print(f"Aggregating over {nodes_matrix.shape[0]} products x {nodes_matrix.shape[1]} windows.")

# ── Mean (and dispersion) nodes per window across products ────────────────
mean_nodes = nodes_matrix.mean(axis=0)
std_nodes = nodes_matrix.std(axis=0)

df_stats = pd.DataFrame({
    'Date': window_dates,
    'MeanNumNodes': mean_nodes,
    'StdNumNodes': std_nodes,
})

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.set_xlabel('Date')
ax1.set_ylabel('Mean Number of Nodes (across products)', color='tab:blue')
ax1.plot(df_stats['Date'], df_stats['MeanNumNodes'], color='tab:blue', alpha=0.9,
         label=f'Mean nodes (n={nodes_matrix.shape[0]} products)')
ax1.fill_between(df_stats['Date'],
                 df_stats['MeanNumNodes'] - df_stats['StdNumNodes'],
                 df_stats['MeanNumNodes'] + df_stats['StdNumNodes'],
                 color='tab:blue', alpha=0.15, label='±1 std')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
fig.autofmt_xdate()

added_christmas = added_month = added_thanksgiving = False
for d in df_stats['Date'].dt.date.unique():
    d_obj = pd.to_datetime(d)
    if d_obj.day == 25 and d_obj.month == 12:
        ax1.axvline(d_obj, color='green', linestyle='--', alpha=0.8,
                    label='Christmas' if not added_christmas else "")
        added_christmas = True
    elif d_obj.month == 11 and d_obj.weekday() == 3 and 22 <= d_obj.day <= 28:
        ax1.axvline(d_obj, color='orange', linestyle='--', alpha=0.8,
                    label='Thanksgiving' if not added_thanksgiving else "")
        added_thanksgiving = True
    elif d_obj.day == 1:
        ax1.axvline(d_obj, color='gray', linestyle=':', alpha=0.4,
                    label='Month Start' if not added_month else "")
        added_month = True

# Highlight peak mean-nodes window
max_idx = int(df_stats['MeanNumNodes'].idxmax())
max_date = df_stats.iloc[max_idx]['Date']
max_val = df_stats.iloc[max_idx]['MeanNumNodes']
ax1.axvline(max_date, color='purple', linestyle='-.', alpha=0.9, label='Max mean-nodes date')
ax1.plot(max_date, max_val, marker='*', markersize=12, color='purple')
ax1.annotate(f'Peak: {max_val:.1f} nodes\n{max_date.strftime("%d/%m/%Y")}',
             xy=(max_date, max_val), xytext=(15, -15), textcoords='offset points',
             color='purple', fontweight='bold',
             arrowprops=dict(arrowstyle="->", color='purple', lw=1.5))

plt.title(f'Mean Nodes per Window across {nodes_matrix.shape[0]} top_12500 products | '
          f'metric={metric}, th={threshold}, window={window_size}, step={step}')
fig.tight_layout()

# ── Save figure ───────────────────────────────────────────────────────────
_BASE_DIR = os.path.join(_HERE, "OverallDistributions")
if is_distance:
    _SAVE_DIR = os.path.join(_BASE_DIR, f"distance_{metric}", "threshold", f"W{window_size}_S{step}_th{threshold}")
else:
    _SAVE_DIR = os.path.join(_BASE_DIR, f"similarity_{metric}", "threshold", f"W{window_size}_S{step}_th{threshold}")
os.makedirs(_SAVE_DIR, exist_ok=True)
_SAVE_PATH = os.path.join(_SAVE_DIR, "mean_nodes_per_window.png")
plt.savefig(_SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"Saved plot to {_SAVE_PATH}")

plt.show()
