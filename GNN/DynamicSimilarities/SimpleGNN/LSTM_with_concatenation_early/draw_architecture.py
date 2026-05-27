"""
Architecture diagram for the GCN + LSTM forecaster with **EARLY
concatenation fusion** (`SimpleGNNLSTMForecaster` in ``gnn_lstm_pyg.py``
inside this folder).

Difference vs the late-concat variant (LSTM_with_concatenation2):
    Late : LSTM(ts_seq) → h_T ; fused = [LN(z) ‖ LN(h_T)] → head
    Early: ts_seq' = [ts_seq ‖ LN(z) broadcast across L] ;
           LSTM(ts_seq') → h_T → head(h_T)

So the graph context `z` influences EVERY LSTM gate at EVERY timestep,
not just the final readout.

Per-step pipeline (model predicts 1 step, recursive over the horizon):

    ego_graph ─► GCNConv₁ → ReLU+Dropout → GCNConv₂ ─► z = h[target]
                                                       (B, out_ch)
                       LayerNorm(z)
                            │
                            ▼  broadcast across L
                       (B, L, out_ch)
                            │
                            ▼  concat with ts_seq along feat axis
    ts_seq (B, L, 1+cal) ──►(B, L, 1+cal+out_ch)──► LSTM (zero h₀, c₀)
                                                     │
                                                     ▼ h_T = lstm_out[:, -1, :]
                                                     │
                                          Linear → ReLU → Dropout → Linear
                                                     │
                                                     ▼
                                          ŷ_(t+1)  (B, 1, 1)
                                          (recursive feedback into ego-graph)

Outputs PNG + PDF next to this file.
"""

import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches


# ----- Hyper-parameters mirrored from main2.py / train.py defaults --------
WINDOW            = 15
LOOKBACK          = 30
HORIZON           = 152
MODEL_HORIZON     = 1
N_NODE_STATS      = 7
N_NODE_CAL        = 21
N_EXOG_LSTM       = 31
GCN_HIDDEN        = 32
GCN_OUT           = 16
LSTM_LAYERS       = 1
LSTM_HIDDEN       = 64
HEAD_HIDDEN       = 64
NEIGHBORS         = 4

IN_CH            = N_NODE_STATS + N_NODE_CAL      # 28
TS_FEAT          = 1 + N_EXOG_LSTM                # 32 (ts + cal exog)
LSTM_INPUT       = TS_FEAT + GCN_OUT              # 32 + 16 = 48 (after early concat)


# ----- Style helpers ------------------------------------------------------
COL_INPUT   = "#E3F2FD"
COL_GCN     = "#C8E6C9"
COL_Z       = "#FFE082"
COL_NORM    = "#FFF3E0"
COL_BCAST   = "#FFCC80"
COL_CONCAT  = "#CE93D8"
COL_LSTM    = "#D1C4E9"
COL_HEAD    = "#FFCDD2"
COL_OUTPUT  = "#B2DFDB"
EDGE        = "#37474F"


def box(ax, xy, w, h, text, color, fontsize=10,
        boxstyle="round,pad=0.05,rounding_size=0.12"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=boxstyle,
        linewidth=1.2, edgecolor=EDGE, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, color="#111")
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, text=None, rad=0.0, color=EDGE, fontsize=8,
          lw=1.4, ls="-"):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="->", mutation_scale=14,
        color=color, linewidth=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, text, ha="center", va="bottom",
                fontsize=fontsize, color="#333",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))


# ----- Figure -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(17, 10.5))
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis("off")

ax.text(10, 12.5,
        "GCN + LSTM Forecaster — EARLY Concatenation Fusion (z broadcast into ts_seq)",
        ha="center", va="center", fontsize=14, fontweight="bold")
ax.text(10, 11.95,
        f"window={WINDOW}   lookback={LOOKBACK}   model_horizon={MODEL_HORIZON} "
        f"(recursive × {HORIZON})   GCN({IN_CH}→{GCN_HIDDEN}→{GCN_OUT})   "
        f"ts_feat={TS_FEAT}  +  z_bcast={GCN_OUT}  →  LSTM_input={LSTM_INPUT}   "
        f"hidden={LSTM_HIDDEN}   head→{HEAD_HIDDEN}",
        ha="center", va="center", fontsize=9.5, color="#555")

# Recursive-loop frame around the per-step pipeline
loop_frame = FancyBboxPatch(
    (0.4, 0.45), 19.2, 11.15,
    boxstyle="round,pad=0.05,rounding_size=0.20",
    linewidth=1.6, edgecolor="#6A1B9A", facecolor="none", linestyle="--",
)
ax.add_patch(loop_frame)
ax.text(0.7, 11.40, f"Recursive inference loop  (repeated {HORIZON} times)",
        ha="left", va="center", fontsize=9, color="#6A1B9A", fontweight="bold")


# === LEFT BRANCH: Ego-graph -> GCN -> z -> LN(z) =========================
ego_cx, ego_cy = 2.4, 8.6
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

ax.text(ego_cx, ego_cy - 2.3,
        f"Ego-graph (target + {NEIGHBORS} neighbours)\n"
        f"rebuilt each step with the latest window\n"
        f"node feats: stats({N_NODE_STATS}) + cal({N_NODE_CAL})  →  x ∈ ℝ^{IN_CH}\n"
        f"edge_weight: similarity score",
        ha="center", va="top", fontsize=9, color="#333")

# GCN layers
box(ax, (5.3, 9.2), 3.2, 0.95,
    f"GCNConv₁\nin={IN_CH}  →  out={GCN_HIDDEN}\nadd_self_loops=True",
    COL_GCN, fontsize=9)
box(ax, (5.5, 8.25), 2.8, 0.55,
    "ReLU  +  Dropout(0.2)", "#E8F5E9", fontsize=8)
box(ax, (5.3, 7.15), 3.2, 0.95,
    f"GCNConv₂\nin={GCN_HIDDEN}  →  out={GCN_OUT}\nadd_self_loops=True",
    COL_GCN, fontsize=9)

arrow(ax, (ego_cx + 0.55, ego_cy), (5.3, 9.67), "x, edge_index, edge_weight")
arrow(ax, (6.9, 9.2), (6.9, 8.80))
arrow(ax, (6.9, 8.25), (6.9, 8.10))

z_c = box(ax, (5.3, 5.85), 3.2, 0.85,
          f"z = h[target]\n(B, {GCN_OUT})",
          COL_Z, fontsize=9)
arrow(ax, (6.9, 7.15), (6.9, 6.70))

ln_c = box(ax, (5.3, 4.65), 3.2, 0.80,
           f"LayerNorm(z)\n(B, {GCN_OUT})",
           COL_NORM, fontsize=9)
arrow(ax, (6.9, 5.85), (6.9, 5.45))

bcast_c = box(ax, (5.3, 3.40), 3.2, 0.80,
              f"broadcast across L={LOOKBACK}\n(B, {LOOKBACK}, {GCN_OUT})",
              COL_BCAST, fontsize=9)
arrow(ax, (6.9, 4.65), (6.9, 4.20))


# === RIGHT BRANCH: ts_seq input ==========================================
ts_in = box(ax, (11.5, 6.40), 5.0, 1.30,
            f"Sequence input  (step t)\n"
            f"ts_seq ∈ ℝ^(B × {LOOKBACK} × {TS_FEAT})\n"
            f"[value_t  ‖  cal_(t+1)]",
            COL_INPUT, fontsize=9)


# === EARLY CONCAT (centre) ===============================================
concat_c = box(ax, (8.0, 3.40), 6.0, 1.10,
               f"Early concat  along feature axis\n"
               f"[ ts_seq  ‖  LN(z)_bcast ]\n"
               f"(B, {LOOKBACK}, {TS_FEAT} + {GCN_OUT} = {LSTM_INPUT})",
               COL_CONCAT, fontsize=10)
# arrows into the early concat
arrow(ax, (8.5, 3.80), (8.0, 3.95), rad=0.0)                 # from broadcast (left)
arrow(ax, (14.0, 6.40), (12.5, 4.50), rad=0.20)              # from ts_seq    (right)


# === LSTM (consumes the augmented sequence) ===============================
lstm_c = box(ax, (8.0, 1.85), 6.0, 1.20,
             f"LSTM  (zero h₀, c₀)\n"
             f"input_size={LSTM_INPUT},  hidden={LSTM_HIDDEN},  layers={LSTM_LAYERS}\n"
             f"batch_first=True   →   h_T = lstm_out[:, -1, :]   (B, {LSTM_HIDDEN})",
             COL_LSTM, fontsize=9)
arrow(ax, (11.0, 3.40), (11.0, 3.05))


# === MLP HEAD (h_T only) =================================================
head_y = 0.55
hb_w   = 2.7
xs     = [4.0, 8.6, 13.2, 17.0]
labels = [
    f"Linear\n{LSTM_HIDDEN} → {HEAD_HIDDEN}\nReLU + Dropout",
    f"Linear\n{HEAD_HIDDEN} → {MODEL_HORIZON}",
    f"ŷ_(t+1)\n(B, {MODEL_HORIZON}, 1)",
]
colors = [COL_LSTM, COL_HEAD, COL_OUTPUT]
xs = xs[: len(labels)]
centers = []
for x, lab, col in zip(xs, labels, colors):
    c = box(ax, (x, head_y), hb_w, 0.75, lab, col, fontsize=9)
    centers.append(c)

# h_T → first head block
arrow(ax, (lstm_c[0], 1.85),
      (centers[0][0] + hb_w / 2 - 0.2, head_y + 0.75),
      rad=-0.25)

# chain head blocks
for a, b in zip(centers[:-1], centers[1:]):
    arrow(ax, (a[0] + hb_w / 2, a[1]), (b[0] - hb_w / 2, b[1]))


# === RECURSIVE FEEDBACK ARROW ============================================
feedback = FancyArrowPatch(
    (centers[-1][0], head_y + 0.75), (ego_cx + 0.55, ego_cy - 0.15),
    arrowstyle="-|>", mutation_scale=16,
    linewidth=1.6, color="#6A1B9A",
    connectionstyle="arc3,rad=-0.55",
)
ax.add_patch(feedback)
ax.text(13.5, 10.45,
        f"append ŷ to lookback  →  rebuild ego-graph  →  step t+1\n"
        f"(repeat {HORIZON} times to produce the full horizon)",
        ha="center", va="center", fontsize=9, color="#6A1B9A",
        bbox=dict(facecolor="#F3E5F5", edgecolor="#6A1B9A",
                  boxstyle="round,pad=0.3"))


# === Legend ==============================================================
legend_patches = [
    mpatches.Patch(facecolor=COL_INPUT,  edgecolor=EDGE, label="Input"),
    mpatches.Patch(facecolor=COL_GCN,    edgecolor=EDGE, label="GCN layer"),
    mpatches.Patch(facecolor=COL_Z,      edgecolor=EDGE, label="Target embedding z"),
    mpatches.Patch(facecolor=COL_NORM,   edgecolor=EDGE, label="LayerNorm"),
    mpatches.Patch(facecolor=COL_BCAST,  edgecolor=EDGE, label="Broadcast"),
    mpatches.Patch(facecolor=COL_CONCAT, edgecolor=EDGE, label="Early concat"),
    mpatches.Patch(facecolor=COL_LSTM,   edgecolor=EDGE, label="LSTM / head"),
    mpatches.Patch(facecolor=COL_HEAD,   edgecolor=EDGE, label="Forecast head"),
    mpatches.Patch(facecolor=COL_OUTPUT, edgecolor=EDGE, label="Output"),
]
ax.legend(handles=legend_patches, loc="lower left",
          bbox_to_anchor=(0.005, 0.005), frameon=True, fontsize=8, ncol=5)


# ----- Save ---------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(out_dir, "gcn_lstm_early_concat_architecture.png")
pdf_path = os.path.join(out_dir, "gcn_lstm_early_concat_architecture.pdf")
plt.tight_layout()
plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
