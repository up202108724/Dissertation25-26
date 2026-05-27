"""
Architecture diagram for the GCN + MLP (improved-fusion) forecaster
(`GCNMLPForecaster`) defined in ``gnn_pyg.py``.

Per-step pipeline (the network itself predicts 1 step at a time and is wrapped
in a recursive inference loop, see ``recursive_inference_gcn_mlp`` in
``gnninference.py``):

    Ego-graph (target + similar items)        <─────────────────┐
        |                                                       |
        v                                                       |
    GCNConv1 → ReLU + Dropout → GCNConv2                        | recursive loop
        |                                                       | (152 steps)
        v                                                       |
    z = h[target]  (B, gcn_out)  ─►  LayerNorm(z)               |
                                            \\                  |
    ts_seq (B, L, 1+cal_dim) ─► flatten ─► Linear(ts_proj)      |
                                            ReLU                |
                                            LayerNorm(ts)       |
                                            /                   |
                              Concatenate [z_norm || ts_norm]   |
                                            |                   |
                                            v                   |
                                MLP (256 → 128 → 1)             |
                                            |                   |
                                            v                   |
                      ŷ_(t+1) (B, 1, 1) ────────────────────────┘
                  (appended to lookback, ego-graph rebuilt for step t+1)
"""

import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches


# ----- Hyper-parameters mirrored from main.py / train.py defaults ---------
WINDOW         = 30          # ego-graph time-series window (= lookback_window)
LOOKBACK       = 30          # MLP lookback
HORIZON        = 152         # full recursive forecast horizon
MODEL_HORIZON  = 1           # the network itself predicts 1 step
N_EXOG         = 21          # |EXOG_COLS|
GCN_HIDDEN     = 32
GCN_OUT        = 64
TS_PROJ_DIM    = 64
MLP_HIDDEN     = (256, 128)
NEIGHBORS      = 4           # purely illustrative

IN_CH      = WINDOW                       # NODE_FEATURES = ['ts'] → dim = window
TS_INPUT   = LOOKBACK * (1 + N_EXOG)      # 30 * 22 = 660
CONCAT_DIM = GCN_OUT + TS_PROJ_DIM        # 128


# ----- Style helpers ------------------------------------------------------
COL_INPUT   = "#E3F2FD"
COL_GCN     = "#C8E6C9"
COL_Z       = "#FFE082"
COL_PROJ    = "#FFE0B2"
COL_NORM    = "#FFF3E0"
COL_CONCAT  = "#F3E5F5"
COL_MLP     = "#B3E5FC"
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
fig, ax = plt.subplots(figsize=(17, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis("off")

ax.text(10, 12.4,
        "GCN + MLP Forecaster (GCNMLPForecaster, improved fusion) — Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold")
ax.text(10, 11.85,
        f"window={WINDOW}   lookback={LOOKBACK}   model_horizon={MODEL_HORIZON} "
        f"(recursive × {HORIZON})   GCN({IN_CH}→{GCN_HIDDEN}→{GCN_OUT})   "
        f"ts_proj={TS_INPUT}→{TS_PROJ_DIM}   MLP{MLP_HIDDEN}",
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


# === LEFT BRANCH: Ego-graph -> GCN -> z ==================================
ego_cx, ego_cy = 2.4, 8.4
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
        f"node feats: ['ts']  →  x ∈ ℝ^{IN_CH}\n"
        f"edge_weight: similarity score",
        ha="center", va="top", fontsize=9, color="#333")

# GCN layers
box(ax, (5.3, 9.0), 3.2, 0.95,
    f"GCNConv₁\nin={IN_CH}  →  out={GCN_HIDDEN}\nadd_self_loops=True",
    COL_GCN, fontsize=9)
box(ax, (5.5, 8.05), 2.8, 0.55,
    "ReLU  +  Dropout(0.2)", "#E8F5E9", fontsize=8)
box(ax, (5.3, 6.95), 3.2, 0.95,
    f"GCNConv₂\nin={GCN_HIDDEN}  →  out={GCN_OUT}\nadd_self_loops=True",
    COL_GCN, fontsize=9)

arrow(ax, (ego_cx + 0.55, ego_cy), (5.3, 9.47), "x, edge_index, edge_weight")
arrow(ax, (6.9, 9.0), (6.9, 8.60))
arrow(ax, (6.9, 8.05), (6.9, 7.90))

z_center = box(ax, (5.3, 5.65), 3.2, 0.85,
               f"z = h[target]\n(B, {GCN_OUT})",
               COL_Z, fontsize=9)
arrow(ax, (6.9, 6.95), (6.9, 6.50))

# z LayerNorm
z_norm = box(ax, (5.3, 4.30), 3.2, 0.85,
             f"LayerNorm(z)\n(B, {GCN_OUT})",
             COL_NORM, fontsize=9)
arrow(ax, (6.9, 5.65), (6.9, 5.15))


# === RIGHT BRANCH: ts_seq → flatten → Linear proj → LayerNorm ============
ts_box = box(ax, (11.5, 9.30), 5.0, 1.30,
             f"Sequence input  (step t)\n"
             f"ts_seq ∈ ℝ^(B × {LOOKBACK} × {1 + N_EXOG})\n"
             f"target (1) + exog ({N_EXOG})",
             COL_INPUT, fontsize=9)

flat_box = box(ax, (12.0, 7.95), 4.0, 0.85,
               f"flatten\n(B, {TS_INPUT})",
               "#E1F5FE", fontsize=9)
arrow(ax, (14.0, 9.30), (14.0, 8.80))

proj_box = box(ax, (12.0, 6.55), 4.0, 0.95,
               f"ts_proj  +  ReLU\n"
               f"Linear({TS_INPUT} → {TS_PROJ_DIM})",
               COL_PROJ, fontsize=9)
arrow(ax, (14.0, 7.95), (14.0, 7.50))

ts_norm = box(ax, (12.0, 5.15), 4.0, 0.85,
              f"LayerNorm(ts)\n(B, {TS_PROJ_DIM})",
              COL_NORM, fontsize=9)
arrow(ax, (14.0, 6.55), (14.0, 6.00))


# === CONCAT ==============================================================
concat_box = box(ax, (5.6, 3.00), 10.4, 0.95,
                 f"Concatenate  [ LayerNorm(z)  ‖  LayerNorm(ts_proj) ]\n"
                 f"(B, {GCN_OUT} + {TS_PROJ_DIM} = {CONCAT_DIM})",
                 COL_CONCAT, fontsize=9)
arrow(ax, (6.9, 4.30), (8.0, 3.95), rad=-0.15)
arrow(ax, (14.0, 5.15), (13.0, 3.95), rad=0.15)


# === MLP =================================================================
mlp_y = 1.85
mlp_w = 3.1
xs = [0.9, 4.5, 8.1, 11.7, 15.3]
labels = (
    [f"Linear\n{CONCAT_DIM} → {MLP_HIDDEN[0]}\nReLU + Dropout"] +
    [f"Linear\n{MLP_HIDDEN[i-1]} → {MLP_HIDDEN[i]}\nReLU + Dropout"
     for i in range(1, len(MLP_HIDDEN))] +
    [f"Linear\n{MLP_HIDDEN[-1]} → {MODEL_HORIZON}"] +
    [f"ŷ_(t+1)\n(B, {MODEL_HORIZON}, 1)"]
)
colors = [COL_MLP] * len(MLP_HIDDEN) + [COL_HEAD, "#B2DFDB"]
xs = xs[: len(labels)]
centers = []
for x, lab, col in zip(xs, labels, colors):
    c = box(ax, (x, mlp_y), mlp_w, 1.00, lab, col, fontsize=9)
    centers.append(c)

# from concat → first MLP block
arrow(ax, (concat_box[0], 3.00), (centers[0][0] + mlp_w / 2 - 0.2, mlp_y + 1.00),
      rad=-0.20)

# chain MLP blocks
for a, b in zip(centers[:-1], centers[1:]):
    arrow(ax, (a[0] + mlp_w / 2, a[1]), (b[0] - mlp_w / 2, b[1]))


# === RECURSIVE FEEDBACK ARROW ============================================
feedback = FancyArrowPatch(
    (centers[-1][0], mlp_y + 1.00), (ego_cx + 0.55, ego_cy - 0.15),
    arrowstyle="-|>", mutation_scale=16,
    linewidth=1.6, color="#6A1B9A",
    connectionstyle="arc3,rad=-0.45",
)
ax.add_patch(feedback)
ax.text(13.0, 0.95,
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
    mpatches.Patch(facecolor=COL_PROJ,   edgecolor=EDGE, label="ts projection"),
    mpatches.Patch(facecolor=COL_NORM,   edgecolor=EDGE, label="LayerNorm"),
    mpatches.Patch(facecolor=COL_CONCAT, edgecolor=EDGE, label="Concatenate"),
    mpatches.Patch(facecolor=COL_MLP,    edgecolor=EDGE, label="MLP block"),
    mpatches.Patch(facecolor=COL_HEAD,   edgecolor=EDGE, label="Forecast head"),
    mpatches.Patch(facecolor=COL_OUTPUT, edgecolor=EDGE, label="Output"),
]
ax.legend(handles=legend_patches, loc="upper right",
          bbox_to_anchor=(0.995, 0.995), frameon=True, fontsize=8, ncol=3)


# ----- Save ---------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(out_dir, "gcn_mlp_improved_architecture.png")
pdf_path = os.path.join(out_dir, "gcn_mlp_improved_architecture.pdf")
plt.tight_layout()
plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
