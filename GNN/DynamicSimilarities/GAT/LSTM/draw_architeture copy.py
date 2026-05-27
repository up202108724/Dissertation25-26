"""
Architecture diagram for the GAT + LSTM forecaster (`GATLSTMForecaster`)
defined in ``gat_lstm_pyg.py``.

Per-step pipeline (the network itself predicts 1 step at a time and is wrapped
in a recursive inference loop, see ``recursive_inference_gat_lstm`` in
``gatinference.py``):

    Ego-graph (target + similar items)        <─────────────────┐
        |                                                       |
        v                                                       |
    GATConv1 (heads=4, concat) → ELU/ReLU + Dropout             | recursive loop
        |                                                       | (152 steps)
        v                                                       |
    GATConv2 (heads=1)                                          |
        |                                                       |
        v                                                       |
    z = h[target]   (B, gat_out_channels)                       |
        |                                                       |
        +--> h_proj → reshape → (layers, B, lstm_hidden)  ──┐   |
        +--> c_proj → reshape → (layers, B, lstm_hidden)  ──┤   |
                                                            v   |
                                          LSTM(ts_seq, (h0, c0))|
                                                            |   |
                                                  last hidden + Dropout
                                                            |   |
                                                Linear(lstm_hidden → 1)
                                                            |   |
                                                            v   |
                              ŷ_(t+1) (B, 1, 1) ────────────────┘

The model itself predicts ONE step at a time (horizon=1). Full 152-step
forecast is produced by `recursive_inference_gat_lstm` in gatinference.py.
"""

import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches


# ----- Hyper-parameters mirrored from main2.py / train.py defaults --------
WINDOW            = 15          # ego-graph time-series window
LOOKBACK          = 30          # LSTM lookback
HORIZON           = 152         # full recursive forecast horizon
MODEL_HORIZON     = 1           # the network itself predicts 1 step at a time
N_NODE_STATS      = 7
N_NODE_CAL        = 21
N_EXOG_LSTM       = 31
GAT_HIDDEN        = 32          # per-head hidden dim
GAT_HEADS         = 4
GAT_OUT           = 16
LSTM_HIDDEN       = 64
LSTM_LAYERS       = 1
NEIGHBORS         = 4           # purely illustrative

IN_CH        = N_NODE_STATS + N_NODE_CAL           # 28
GAT1_OUT     = GAT_HIDDEN * GAT_HEADS              # 128
LSTM_INPUT   = 1 + N_EXOG_LSTM                     # 32


# ----- Style helpers ------------------------------------------------------
COL_INPUT   = "#E3F2FD"
COL_GAT     = "#FFE0B2"
COL_Z       = "#FFE082"
COL_PROJ    = "#FFCCBC"
COL_LSTM    = "#D1C4E9"
COL_HEAD    = "#FFCDD2"
COL_OUTPUT  = "#F8BBD0"
EDGE        = "#37474F"


def box(ax, xy, w, h, text, color, fontsize=10,
        boxstyle="round,pad=0.05,rounding_size=0.12"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=boxstyle,
        linewidth=1.2, edgecolor=EDGE, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, color="#111")
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, text=None, rad=0.0, color=EDGE, fontsize=8):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.2, color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, text, ha="center", va="bottom",
                fontsize=fontsize, color="#333",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


# ----- Figure -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis("off")

ax.text(10, 12.4, "GAT + LSTM Forecaster (GATLSTMForecaster) — Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold")
ax.text(10, 11.85,
        f"window={WINDOW}   lookback={LOOKBACK}   model_horizon={MODEL_HORIZON} "
        f"(recursive × {HORIZON})   "
        f"GAT(heads={GAT_HEADS}, {IN_CH}→{GAT_HIDDEN}·{GAT_HEADS}={GAT1_OUT}→{GAT_OUT})   "
        f"LSTM(input={LSTM_INPUT}, hidden={LSTM_HIDDEN}, layers={LSTM_LAYERS})",
        ha="center", va="center", fontsize=10, color="#555")

# Recursive-loop frame around the per-step pipeline
loop_frame = FancyBboxPatch(
    (0.4, 0.45), 19.2, 11.05,
    boxstyle="round,pad=0.05,rounding_size=0.20",
    linewidth=1.6, edgecolor="#6A1B9A", facecolor="none", linestyle="--",
)
ax.add_patch(loop_frame)
ax.text(0.7, 11.30, f"Recursive inference loop  (repeated {HORIZON} times)",
        ha="left", va="center", fontsize=9, color="#6A1B9A", fontweight="bold")


# === LEFT BRANCH: Ego-graph -> GAT -> z ==================================
ego_cx, ego_cy = 2.5, 8.0
target = Circle((ego_cx, ego_cy), 0.45, facecolor="#90CAF9",
                edgecolor=EDGE, linewidth=1.5, zorder=3)
ax.add_patch(target)
ax.text(ego_cx, ego_cy, "T", ha="center", va="center",
        fontsize=11, fontweight="bold", zorder=4)

for k in range(NEIGHBORS):
    ang = math.pi / 2 + 2 * math.pi * (k + 1) / (NEIGHBORS + 1)
    nx_, ny_ = ego_cx + 1.45 * math.cos(ang), ego_cy + 1.05 * math.sin(ang)
    c = Circle((nx_, ny_), 0.30, facecolor="#FFCC80",
               edgecolor=EDGE, linewidth=1.0, zorder=3)
    ax.add_patch(c)
    ax.text(nx_, ny_, f"n{k+1}", ha="center", va="center", fontsize=8, zorder=4)
    ax.plot([ego_cx, nx_], [ego_cy, ny_], color=EDGE, linewidth=0.9, zorder=2)

ax.text(ego_cx, ego_cy - 2.2,
        f"Ego-graph (target + {NEIGHBORS} neighbours)\n"
        f"rebuilt each step with the latest window\n"
        f"node feats: stats({N_NODE_STATS}) + cal({N_NODE_CAL})  →  x ∈ ℝ^{IN_CH}\n"
        f"edge_attr: similarity score (edge_dim=1)",
        ha="center", va="top", fontsize=9, color="#333")

# GAT layers
box(ax, (5.4, 8.6), 3.4, 0.95,
    f"GATConv₁\nin={IN_CH}  →  out={GAT_HIDDEN}·{GAT_HEADS} = {GAT1_OUT}\n"
    f"heads={GAT_HEADS} (concat), edge_dim=1",
    COL_GAT, fontsize=9)
box(ax, (5.6, 7.65), 3.0, 0.55,
    "ELU  +  Dropout(0.2)", "#FFF3E0", fontsize=8)
box(ax, (5.4, 6.55), 3.4, 0.95,
    f"GATConv₂\nin={GAT1_OUT}  →  out={GAT_OUT}\n"
    f"heads=1, edge_dim=1",
    COL_GAT, fontsize=9)

arrow(ax, (ego_cx + 0.55, ego_cy), (5.4, 9.07), "x, edge_index, edge_attr")
arrow(ax, (7.1, 8.6), (7.1, 8.20))
arrow(ax, (7.1, 7.65), (7.1, 7.50))

# select target embedding z
z_center = box(ax, (9.4, 6.75), 2.4, 0.85,
               f"z = h[target]\n(B, {GAT_OUT})",
               COL_Z, fontsize=9)
arrow(ax, (8.8, 7.02), (9.4, 7.17))


# === MIDDLE: projection of z to (h0, c0) =================================
h_proj = box(ax, (12.5, 8.10), 3.6, 0.95,
             f"h₀ projection\n"
             f"Linear({GAT_OUT} → {LSTM_HIDDEN}·{LSTM_LAYERS})  →  reshape\n"
             f"({LSTM_LAYERS}, B, {LSTM_HIDDEN})",
             COL_PROJ, fontsize=9)
c_proj = box(ax, (12.5, 6.80), 3.6, 0.95,
             f"c₀ projection\n"
             f"Linear({GAT_OUT} → {LSTM_HIDDEN}·{LSTM_LAYERS})  →  reshape\n"
             f"({LSTM_LAYERS}, B, {LSTM_HIDDEN})",
             COL_PROJ, fontsize=9)
arrow(ax, (11.8, 7.30), (12.5, 8.57), "z", rad=0.20)
arrow(ax, (11.8, 7.10), (12.5, 7.27), "z", rad=-0.10)


# === RIGHT INPUT: ts_seq =================================================
ts_box = box(ax, (16.5, 9.40), 3.3, 1.40,
             f"Sequence input  (step t)\n"
             f"ts_seq ∈ ℝ^(B × {LOOKBACK} × {LSTM_INPUT})\n"
             f"[value_t  ‖  cal_(t+1)]",
             COL_INPUT, fontsize=9)


# === LSTM ================================================================
lstm_box = box(ax, (16.3, 6.40), 3.5, 1.45,
               f"LSTM\n"
               f"input_size={LSTM_INPUT},  hidden={LSTM_HIDDEN}\n"
               f"num_layers={LSTM_LAYERS},  batch_first=True\n"
               f"init: (h₀, c₀) from GAT z",
               COL_LSTM, fontsize=9)
arrow(ax, (18.0, 9.40), (18.0, 7.85), "ts_seq")
arrow(ax, (16.1, 8.57), (16.3, 7.55), "h₀", rad=-0.10)
arrow(ax, (16.1, 7.27), (16.3, 6.95), "c₀", rad=-0.10)


# === Head (1 step) =======================================================
last_box = box(ax, (16.3, 4.40), 3.5, 0.95,
               f"Last hidden  +  Dropout\n"
               f"Linear({LSTM_HIDDEN} → {MODEL_HORIZON})",
               COL_HEAD, fontsize=9)
arrow(ax, (18.0, 6.40), (18.0, 5.35))

step_out = box(ax, (16.3, 2.70), 3.5, 0.80,
               f"ŷ_(t+1)\n(B, {MODEL_HORIZON}, 1)",
               "#B2DFDB", fontsize=9)
arrow(ax, (18.0, 4.40), (18.0, 3.50))


# === RECURSIVE FEEDBACK ARROW ============================================
feedback = FancyArrowPatch(
    (16.3, 3.10), (ego_cx + 0.55, ego_cy + 0.15),
    arrowstyle="-|>", mutation_scale=16,
    linewidth=1.6, color="#6A1B9A",
    connectionstyle="arc3,rad=-0.35",
)
ax.add_patch(feedback)
ax.text(9.5, 2.05,
        f"append ŷ to lookback  →  rebuild ego-graph  →  step t+1\n"
        f"(repeat {HORIZON} times to produce the full horizon)",
        ha="center", va="center", fontsize=9, color="#6A1B9A",
        bbox=dict(facecolor="#F3E5F5", edgecolor="#6A1B9A",
                  boxstyle="round,pad=0.3"))


# === FINAL CONCATENATED FORECAST =========================================
out_box = box(ax, (0.8, 0.65), 5.6, 1.10,
              f"Final forecast  (after loop)\n(B, {HORIZON}, 1)",
              COL_OUTPUT, fontsize=10)


# === Legend ==============================================================
legend_patches = [
    mpatches.Patch(facecolor=COL_INPUT,  edgecolor=EDGE, label="Input"),
    mpatches.Patch(facecolor=COL_GAT,    edgecolor=EDGE, label="GAT layer"),
    mpatches.Patch(facecolor=COL_Z,      edgecolor=EDGE, label="Target embedding z"),
    mpatches.Patch(facecolor=COL_PROJ,   edgecolor=EDGE, label="Init projection"),
    mpatches.Patch(facecolor=COL_LSTM,   edgecolor=EDGE, label="LSTM"),
    mpatches.Patch(facecolor=COL_HEAD,   edgecolor=EDGE, label="Forecast head"),
    mpatches.Patch(facecolor=COL_OUTPUT, edgecolor=EDGE, label="Output"),
]
ax.legend(handles=legend_patches, loc="lower left",
          bbox_to_anchor=(0.005, 0.005), frameon=True, fontsize=8, ncol=4)


# ----- Save ---------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(out_dir, "gat_lstm_architecture.png")
pdf_path = os.path.join(out_dir, "gat_lstm_architecture.pdf")
plt.tight_layout()
plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
