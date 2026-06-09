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
THRESHOLDS = [0.70, 0.65, 0.60]
COLORS     = ['#1f77b4', '#ff7f0e', '#2ca02c']   # blue, orange, green
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

# ── Compute mean nodes per window for each threshold ─────────────────────
results = {}   # threshold → (window_dates, mean_nodes, std_nodes)

for threshold in THRESHOLDS:
    print(f"\n=== Threshold {threshold} ===")
    nodes_per_product = []
    window_dates      = None

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
        if window_dates is None:
            window_dates = pd.to_datetime([g.graph['end_date'] for g in graphs])

    print()
    min_len      = min(len(a) for a in nodes_per_product)
    nodes_matrix = np.vstack([a[:min_len] for a in nodes_per_product])
    window_dates = window_dates[:min_len]

    results[threshold] = (
        window_dates,
        nodes_matrix.mean(axis=0),
        nodes_matrix.std(axis=0),
    )
    print(f"  → {nodes_matrix.shape[0]} products × {nodes_matrix.shape[1]} windows")

# ── Combined plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

for (threshold, (dates, mean_n, std_n)), color in zip(results.items(), COLORS):
    ax.plot(dates, mean_n, color=color, alpha=0.9,
            label=fr'$\tau={threshold}$', linewidth=1.4)
    ax.fill_between(dates, mean_n - std_n, mean_n + std_n,
                    color=color, alpha=0.10)

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Mean Number of Nodes (across products)', fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
fig.autofmt_xdate()

# Seasonal markers (use dates from any threshold — they are the same)
ref_dates = results[THRESHOLDS[0]][0]
added_christmas = added_thanksgiving = added_month = False
for d in pd.Series(ref_dates).dt.date.unique():
    d_obj = pd.to_datetime(d)
    if d_obj.day == 25 and d_obj.month == 12:
        ax.axvline(d_obj, color='green', linestyle='--', alpha=0.7,
                   label='Christmas' if not added_christmas else "")
        added_christmas = True
    elif d_obj.month == 11 and d_obj.weekday() == 3 and 22 <= d_obj.day <= 28:
        ax.axvline(d_obj, color='orange', linestyle='--', alpha=0.7,
                   label='Thanksgiving' if not added_thanksgiving else "")
        added_thanksgiving = True
    elif d_obj.day == 1:
        ax.axvline(d_obj, color='gray', linestyle=':', alpha=0.35,
                   label='Month start' if not added_month else "")
        added_month = True

ax.legend(fontsize=10)
plt.title(
    f'Mean Nodes per Window — Spearman | '
    f'window={window_size}, step={step} | '
    f'thresholds: {THRESHOLDS}',
    fontsize=11,
)
fig.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.join(_HERE, "OverallDistributions")
_SAVE_DIR = os.path.join(_BASE_DIR, f"similarity_{metric}", "threshold",
                         f"W{window_size}_S{step}_combined_thresholds")
os.makedirs(_SAVE_DIR, exist_ok=True)
_SAVE_PATH = os.path.join(_SAVE_DIR, "mean_nodes_combined_thresholds.png")
plt.savefig(_SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved combined plot to {_SAVE_PATH}")
plt.show()
