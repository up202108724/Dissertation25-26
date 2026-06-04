import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

# Make sure the GlobalModel utils and graph_plot are importable
NOTEBOOK_DIR = Path(os.getcwd())
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from utils import compute_similarities_allvsall
from graph_plot import plot_networkx_plotly

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH  = str(NOTEBOOK_DIR / '../../dataset/data_smooth_erratic.feather')
DATE_COL   = 'date'
TARGET_COL = 'value'

METRIC               = 'spearman'    # 'pearson' | 'spearman' | 'kendall'
SIMILARITY_THRESHOLD = 0.6        # only edges with sim >= threshold are kept


# ── Load dataset & build wide pivot (item_id × date) ──────────────────────
df = pd.read_feather(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values([DATE_COL, 'item_id']).reset_index(drop=True)

df_wide = (
    df.pivot_table(index='item_id', columns=DATE_COL, values=TARGET_COL, aggfunc='sum')
    .fillna(0)
)

# Optional category labels for colouring nodes in the plot
cat_labels_dict = (
    df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()
    if 'cat_label' in df.columns else {}
)

item_ids = df_wide.index.tolist()
print(f"Loaded: {len(item_ids)} products × {df_wide.shape[1]} time steps")

# ── Compute all-vs-all similarity matrix (GPU-accelerated if available) ────
all_ts = df_wide.values.astype(np.float32)   # (N, T)

print(f"Computing {METRIC} similarity for {len(item_ids)} × {len(item_ids)} pairs...")
sim_matrix = compute_similarities_allvsall(all_ts, metric=METRIC)

print(f"Done.  Matrix shape: {sim_matrix.shape}")
print(f"Value range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")


# ── Apply threshold → keep only connected products ─────────────────────────
N = len(item_ids)

# Vectorised: find all upper-triangle pairs where sim >= threshold
mask = np.triu(sim_matrix >= SIMILARITY_THRESHOLD, k=1)  # exclude self-loops
rows, cols = np.where(mask)

G_full = nx.Graph()
G_full.add_nodes_from(item_ids)
for i, j in zip(rows, cols):
    G_full.add_edge(item_ids[i], item_ids[j], weight=float(sim_matrix[i, j]))

# Sub-graph: only nodes with at least one edge
connected_nodes = [n for n, d in G_full.degree() if d > 0]
G = G_full.subgraph(connected_nodes).copy()

# Attach category label to each node (used by plot_networkx_plotly for colouring)
for node in G.nodes():
    G.nodes[node]['cat_label'] = cat_labels_dict.get(node, 'Unknown')

isolated = N - G.number_of_nodes()
print(f"Threshold = {SIMILARITY_THRESHOLD}  |  metric = {METRIC}")
print(f"  Connected products : {G.number_of_nodes():>5d}  ({isolated} isolated, removed)")
print(f"  Edges              : {G.number_of_edges():>5d}")

