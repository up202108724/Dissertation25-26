"""
Training-logic architecture for the GAT + LSTM forecaster
(see train.py :: train_gat_lstm and main2.py).

The model itself is drawn in ``draw_architeture.py``. Here we visualise how a
sequence of ego-graphs is built from the multi-product panel and how each
sliding window is turned into a training sample (y, graph, ts_seq) consumed
by ``GATLSTMForecaster``.

Pipeline
--------
                  Wide multi-product panel (T × P)
                              │
                              ▼
              compute_similarities_1vsAll  (spearman, window W)
                              │
                              ▼
              neighbourhood_graph  ─── one ego-graph per time-step
                              │              g_0, g_1, …, g_{T-1}
                              ▼
        ┌───────── Train / Val / Test temporal split ─────────┐
        │                                                     │
        ▼                                                     ▼
   scale target with MinMaxScaler.fit on train, transform on val/test
        │
        ▼
   make_single_windows( ts, cal, lookback=L, horizon=1, graphs )
       per index i ∈ [0, T − L − 1]:
         y_i      = target[i+L : i+L+1]                       (1, 1)
         graph_i  = padded_graphs[i+L−1]  with node features
                    recomputed by generate_node_features
                    (stats + cal_next + optional cal_lookback)
         ts_seq_i = [(value_t, cal_{t+1})] for t = 0..L−1     (L, 1+cal_dim)
        │
        ▼
   SingleGraphDataset  →  DataLoader( batch=B, single_graph_collate )
                                │
                                ▼
                    (y_batch, pyg_Batch, ts_batch)
                                │
                                ▼
                   GATLSTMForecaster(pyg_batch, ptr, ts_batch)
                                │
                                ▼
                    loss_fn(pred, y) → AdamW step
                    + grad-clip 1.0  + early-stopping on val
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches


# ----- Hyper-parameters mirrored from main2.py / TrainConfig --------------
LOOKBACK         = 30
HORIZON          = 1            # model horizon (recursive ×152 at inference)
GRAPH_WINDOW     = 15
BATCH_SIZE       = 32
SPEARMAN_THR     = "{0.75, 0.82, 0.85, 0.88, 0.91}"
LR               = "1e-4"
WD               = "1e-3"
EPOCHS           = 30
PATIENCE         = 150


# ----- Colours ------------------------------------------------------------
COL_DATA     = "#E3F2FD"
COL_SIM      = "#FFE0B2"
COL_GRAPH    = "#C8E6C9"
COL_SPLIT    = "#FFF59D"
COL_SCALE    = "#F8BBD0"
COL_WIN      = "#D1C4E9"
COL_LOADER   = "#B2DFDB"
COL_FORWARD  = "#FFCDD2"
COL_LOSS     = "#FFCCBC"
COL_NOTE     = "#FFFFFF"
EDGE         = "#37474F"


def box(ax, xy, w, h, text, color, fontsize=9,
        boxstyle="round,pad=0.05,rounding_size=0.12", edgecolor=EDGE, lw=1.2):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                       linewidth=lw, edgecolor=edgecolor, facecolor=color)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, color="#111")
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, text=None, rad=0.0, color=EDGE, fontsize=8, lw=1.2):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                        linewidth=lw, color=color,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + 0.05, my + 0.10, text, ha="center", va="bottom",
                fontsize=fontsize, color="#333",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


# ----- Figure -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(17, 11.5))
ax.set_xlim(0, 20)
ax.set_ylim(0, 15)
ax.axis("off")

ax.text(10, 14.4, "GAT + LSTM — Training Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold")
ax.text(10, 13.9,
        f"sliding ego-graphs (window={GRAPH_WINDOW})  →  per-step samples  "
        f"(lookback={LOOKBACK}, horizon={HORIZON})  →  batched training",
        ha="center", va="center", fontsize=10, color="#555")


# === STAGE 1: data + similarity + graph sequence =========================
data_c = box(ax, (0.5, 11.4), 4.2, 1.40,
             "Wide multi-product panel\n(T days × P products)\n"
             "df_wide_global  +  EXOG_COLS",
             COL_DATA, fontsize=9)

sim_c = box(ax, (5.5, 11.4), 4.6, 1.40,
            "compute_similarities_1vsAll\n"
            "Spearman over sliding window\n"
            f"window={GRAPH_WINDOW}, step=1, train-end cutoff",
            COL_SIM, fontsize=9)

graph_c = box(ax, (10.9, 11.4), 4.7, 1.40,
              "neighbourhood_graph\n"
              "→ one PyG ego-graph per timestep\n"
              f"g₀, g₁, …, g_(T−1)  (threshold ∈ {SPEARMAN_THR})",
              COL_GRAPH, fontsize=9)

note_c = box(ax, (16.1, 11.4), 3.5, 1.40,
             "Each graph:\n"
             "• target (central) node\n"
             "• k similar items above τ\n"
             "• edge_attr = similarity score",
             COL_NOTE, fontsize=8, edgecolor="#999")

arrow(ax, (4.7, 12.10), (5.5, 12.10))
arrow(ax, (10.1, 12.10), (10.9, 12.10))
arrow(ax, (15.6, 12.10), (16.1, 12.10), color="#999")


# === STAGE 2: temporal split + scaling ===================================
split_c = box(ax, (0.5, 9.20), 8.8, 1.50,
              "Temporal split  (per product)\n"
              "train  [0 : val_start)   |   val  [val_start : test_start)   |   test  [test_start : T)\n"
              f"val_size, test_size = forecast_horizon (152)",
              COL_SPLIT, fontsize=9)

scale_c = box(ax, (9.7, 9.20), 9.9, 1.50,
              "MinMaxScaler (per product)\n"
              "scaler.fit_transform(train_target)   →   scaler.transform(val_target / test_target)\n"
              "calendar/exogenous columns NOT rescaled (already binary / cyclic)",
              COL_SCALE, fontsize=9)

arrow(ax, (5.0, 11.4), (5.0, 10.70))
arrow(ax, (13.3, 11.4), (13.3, 10.70))
arrow(ax, (9.3, 9.95), (9.7, 9.95))


# === STAGE 3: sliding-window sample construction =========================
win_c = box(ax, (0.5, 5.90), 19.1, 2.80,
            "make_single_windows( ts, cal, lookback=L, horizon=1, graphs, graph_window_size=15 )\n"
            "for each sliding index i ∈ [0, T − L − 1]:\n"
            f"   y_i       = target[i+L : i+L+1]                                     shape (1, 1)\n"
            "   graph_i   = padded_graphs[i + L − 1]\n"
            "              • target-node feats = generate_node_features(target_ts, cal_next, cal_lookback?)\n"
            "              • neighbour feats   = generate_node_features(neighbor_ts, is_neighbor=True)\n"
            f"   ts_seq_i  = [(value_t , cal_(t+1))]  for t = 0..L−1                shape (L, 1+cal_dim)\n"
            "→ N = (T − L) samples each:  (y_i, graph_i, ts_seq_i)",
            COL_WIN, fontsize=9)

arrow(ax, (5.0, 9.20), (5.0, 8.70))
arrow(ax, (14.5, 9.20), (14.5, 8.70))


# === STAGE 4: dataset / dataloader =======================================
ds_c = box(ax, (0.5, 3.80), 9.2, 1.70,
           "SingleGraphDataset(y, graphs, ts_seqs)\n"
           "__getitem__ → (y_i, graph_i, ts_seq_i)",
           COL_LOADER, fontsize=9)

dl_c = box(ax, (10.1, 3.80), 9.5, 1.70,
           "DataLoader(batch_size=B, collate_fn=single_graph_collate)\n"
           "• stack y           → (B, 1, 1)\n"
           "• Batch.from_data_list(graphs)   → pyg_Batch (one big disconnected graph + ptr)\n"
           "• stack ts_seq      → (B, L, 1+cal_dim)",
           COL_LOADER, fontsize=9)

arrow(ax, (5.0, 5.90), (5.0, 5.50))
arrow(ax, (9.7, 4.65), (10.1, 4.65))


# === STAGE 5: forward + loss + opt =======================================
fwd_c = box(ax, (0.5, 1.40), 9.2, 1.90,
            "GATLSTMForecaster(pyg_Batch, ptr[:-1], ts_batch)\n"
            "GAT encodes every ego-graph in the batch\n"
            "→ z = h[central_node] per sample → init (h₀, c₀)\n"
            "→ LSTM(ts_seq)  →  head  →  pred (B, 1, 1)",
            COL_FORWARD, fontsize=9)

loss_c = box(ax, (10.1, 1.40), 9.5, 1.90,
             "Training step\n"
             f"loss = loss_fn(pred, y_batch)        (MSE / MAE / Huber)\n"
             f"opt = AdamW(lr={LR}, weight_decay={WD})    grad_clip = 1.0\n"
             f"epochs={EPOCHS},   early stop after {PATIENCE} epochs w/o val improvement\n"
             "best model → best_models/seed_<S>/<loss>/gnn_lstm_product_<id>.pth",
             COL_LOSS, fontsize=9)

arrow(ax, (5.0, 3.80), (5.0, 3.30))
arrow(ax, (9.7, 2.35), (10.1, 2.35))

# Feedback arrow (back-prop)
back = FancyArrowPatch((14.8, 1.40), (5.0, 1.40),
                       arrowstyle="-|>", mutation_scale=16,
                       linewidth=1.6, color="#6A1B9A",
                       connectionstyle="arc3,rad=0.30")
ax.add_patch(back)
ax.text(10.0, 0.55, "back-prop  →  update GAT, projections, LSTM, head",
        ha="center", va="center", fontsize=9, color="#6A1B9A",
        bbox=dict(facecolor="#F3E5F5", edgecolor="#6A1B9A",
                  boxstyle="round,pad=0.3"))


# === Legend ==============================================================
legend_patches = [
    mpatches.Patch(facecolor=COL_DATA,    edgecolor=EDGE, label="Raw data"),
    mpatches.Patch(facecolor=COL_SIM,     edgecolor=EDGE, label="Similarity computation"),
    mpatches.Patch(facecolor=COL_GRAPH,   edgecolor=EDGE, label="Ego-graph sequence"),
    mpatches.Patch(facecolor=COL_SPLIT,   edgecolor=EDGE, label="Temporal split"),
    mpatches.Patch(facecolor=COL_SCALE,   edgecolor=EDGE, label="Scaling"),
    mpatches.Patch(facecolor=COL_WIN,     edgecolor=EDGE, label="Sliding windows"),
    mpatches.Patch(facecolor=COL_LOADER,  edgecolor=EDGE, label="Dataset / DataLoader"),
    mpatches.Patch(facecolor=COL_FORWARD, edgecolor=EDGE, label="Forward pass"),
    mpatches.Patch(facecolor=COL_LOSS,    edgecolor=EDGE, label="Loss + optimiser"),
]
ax.legend(handles=legend_patches, loc="upper left",
          bbox_to_anchor=(0.005, 0.045), frameon=True, fontsize=8, ncol=5)


# ----- Save ---------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(out_dir, "gat_lstm_training_architecture.png")
pdf_path = os.path.join(out_dir, "gat_lstm_training_architecture.pdf")
plt.tight_layout()
plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
