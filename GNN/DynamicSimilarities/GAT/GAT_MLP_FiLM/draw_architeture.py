"""
Architecture diagram for the GAT + MLP forecaster — RECURSIVE INFERENCE.

The model (`GATMLPForecaster`, see ``gat_pyg.py``) is trained for
one-step-ahead prediction (horizon = 1). Multi-step forecasts over
H_TOTAL = 152 are produced at inference time by
``recursive_inference_gat_mlp`` (see ``gatinference.py``):

    for i in 1..H_TOTAL:
        - rebuild ego-graph from the current (scaled) lookback window
        - GAT → z (target embedding)
        - concat [ z || flatten(ts_seq) ] → MLP → 1-step prediction y_hat
        - unscale, append to forecast, shift lookback & calendar windows

Outputs a PNG (and a PDF) next to this file.
"""

import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches


# ----- Hyper-parameters mirrored from main.py / train.py defaults ---------
WINDOW       = 15           # node-feature time-series window (graph window)
LOOKBACK     = 30           # MLP lookback
HORIZON      = 1            # model output per forward pass (recursive)
H_TOTAL      = 152          # full forecast horizon obtained by recursion
N_EXOG       = 22           # |EXOG_COLS| in main.py
GAT_HIDDEN   = 32
GAT_OUT      = 16
GAT_HEADS    = 4
MLP_HIDDEN   = (256, 128)
NEIGHBORS    = 4            # purely illustrative


# ----- Style helpers ------------------------------------------------------
COL_INPUT   = "#E3F2FD"
COL_GAT     = "#FFE0B2"
COL_CONCAT  = "#F3E5F5"
COL_MLP     = "#C8E6C9"
COL_OUTPUT  = "#FFCDD2"
COL_LOOP    = "#FFF9C4"
EDGE        = "#37474F"
LOOP_EDGE   = "#C62828"


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
    return (x + w / 2, y + h / 2), (x, y, w, h)


def arrow(ax, p1, p2, text=None, rad=0.0, color=EDGE, fontsize=8,
          linestyle="-", linewidth=1.2):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14,
        linewidth=linewidth, color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=linestyle,
    )
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, text, ha="center", va="bottom",
                fontsize=fontsize, color="#333",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


# ----- Figure -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 9.5))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis("off")

ax.text(10, 11.45, "GAT + MLP Forecaster — Recursive Inference",
        ha="center", va="center", fontsize=15, fontweight="bold")
ax.text(10, 10.90,
        f"graph window={WINDOW}   lookback={LOOKBACK}   "
        f"per-step horizon={HORIZON}   total horizon={H_TOTAL}   "
        f"GAT(heads={GAT_HEADS}, {GAT_HIDDEN}->{GAT_OUT})   MLP{MLP_HIDDEN}",
        ha="center", va="center", fontsize=10, color="#555")


# === Recursive-loop frame ===============================================
loop_frame = FancyBboxPatch(
    (0.5, 3.7), 19.0, 6.5,
    boxstyle="round,pad=0.05,rounding_size=0.25",
    linewidth=1.4, edgecolor=LOOP_EDGE, facecolor=COL_LOOP, alpha=0.25,
    linestyle="--",
)
ax.add_patch(loop_frame)
ax.text(0.75, 10.05,
        f"Recursive loop  -  repeated for  i = 1 .. {H_TOTAL}",
        ha="left", va="top", fontsize=10, fontweight="bold", color=LOOP_EDGE)


# === LEFT BRANCH: Ego-graph -> GAT -> z =================================
ego_cx, ego_cy = 2.5, 7.0
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
    ax.text(nx_, ny_, f"n{k+1}", ha="center", va="center",
            fontsize=8, zorder=4)
    ax.plot([ego_cx, nx_], [ego_cy, ny_], color=EDGE, linewidth=0.9, zorder=2)

ax.text(ego_cx, ego_cy - 2.1,
        f"Ego-graph rebuilt each step\n"
        f"from current lookback window\n"
        f"node feats: ['ts']  ->  x in R^{WINDOW}\n"
        f"edge_attr: similarity score (edge_dim=1)",
        ha="center", va="top", fontsize=9, color="#333")

# GAT layers
gat1, _ = box(ax, (5.5, 7.6), 3.2, 0.95,
              f"GATConv1\nin={WINDOW}  ->  out={GAT_HIDDEN}\n"
              f"heads={GAT_HEADS} (avg),  edge_dim=1",
              COL_GAT, fontsize=9)
gat2, _ = box(ax, (5.5, 6.0), 3.2, 0.95,
              f"GATConv2\nin={GAT_HIDDEN}  ->  out={GAT_OUT}\n"
              f"heads=1,  edge_dim=1",
              COL_GAT, fontsize=9)
relu_drop, _ = box(ax, (5.7, 5.05), 2.8, 0.55,
                   "ReLU  +  Dropout(0.2)", "#FFF3E0", fontsize=8)

arrow(ax, (ego_cx + 0.55, ego_cy), (5.5, 8.07), "x, edge_index, edge_attr")
arrow(ax, (7.1, 7.6), (7.1, 6.95))
arrow(ax, (7.1, 6.0), (7.1, 5.60))

# select target embedding z
z_box_center, _ = box(ax, (9.2, 6.10), 2.2, 0.85,
                      f"z = h[target]\n(B, {GAT_OUT})",
                      "#FFE082", fontsize=9)
arrow(ax, (8.7, 6.47), (9.2, 6.52))


# === RIGHT BRANCH: lookback time-series ==================================
ts_box, _ = box(ax, (12.7, 7.7), 5.6, 1.30,
                f"Lookback window  (step i)\n"
                f"ts_seq in R^(B x {LOOKBACK} x {1 + N_EXOG})\n"
                f"target (1) + exog ({N_EXOG})  - shifted each step",
                COL_INPUT, fontsize=9)

flat_box, _ = box(ax, (13.5, 6.10), 4.0, 0.85,
                  f"flatten\n(B, {LOOKBACK * (1 + N_EXOG)})",
                  "#E1F5FE", fontsize=9)
arrow(ax, (15.5, 7.70), (15.5, 6.95))


# === CONCAT =============================================================
concat_box, _ = box(ax, (9.2, 4.30), 8.3, 0.90,
                    f"Concatenate  [ z || flat(ts_seq) ]\n"
                    f"(B, {GAT_OUT} + {LOOKBACK * (1 + N_EXOG)} "
                    f"= {GAT_OUT + LOOKBACK * (1 + N_EXOG)})",
                    COL_CONCAT, fontsize=9)
arrow(ax, (10.3, 6.10), (10.8, 5.20), rad=-0.15)
arrow(ax, (15.5, 6.10), (15.5, 5.20))


# === MLP ================================================================
mlp_y = 2.55
prev_dim = GAT_OUT + LOOKBACK * (1 + N_EXOG)
xs = [3.6, 7.3, 11.0, 14.7]
labels = (
    [f"Linear\n{prev_dim} -> {MLP_HIDDEN[0]}\nReLU + Dropout"] +
    [f"Linear\n{MLP_HIDDEN[i-1]} -> {MLP_HIDDEN[i]}\nReLU + Dropout"
     for i in range(1, len(MLP_HIDDEN))] +
    [f"Linear\n{MLP_HIDDEN[-1]} -> {HORIZON}\n(1-step prediction)"]
)
centers = []
for x, lab in zip(xs, labels):
    c, _ = box(ax, (x, mlp_y), 3.0, 1.05, lab, COL_MLP, fontsize=9)
    centers.append(c)

# from concat to first MLP layer
arrow(ax, (13.35, 4.30), (centers[0][0], mlp_y + 1.05), rad=0.25)
for a, b in zip(centers[:-1], centers[1:]):
    arrow(ax, (a[0] + 1.5, a[1]), (b[0] - 1.5, b[1]))


# === ONE-STEP PREDICTION + FEEDBACK =====================================
yhat_box, _ = box(ax, (8.4, 0.9), 4.0, 1.0,
                  f"y_hat (t+1)  (unscaled)\n(B, {HORIZON}, 1)",
                  COL_OUTPUT, fontsize=10)
arrow(ax, (centers[-1][0], mlp_y), (10.4, 1.90), rad=0.25)

# Final aggregated forecast (outside the loop)
final_box, _ = box(ax, (13.6, 0.7), 5.7, 1.20,
                   f"Aggregated Forecast\n"
                   f"[y_hat(t+1), y_hat(t+2), ..., y_hat(t+{H_TOTAL})]\n"
                   f"(B, {H_TOTAL}, 1)",
                   "#FFEBEE", fontsize=10)
arrow(ax, (12.4, 1.40), (13.6, 1.30))

# === Recursive feedback arrow (y_hat -> lookback) =======================
fb = FancyArrowPatch(
    (8.4, 1.40),
    (12.7, 8.20),
    arrowstyle="-|>", mutation_scale=18,
    linewidth=1.8, color=LOOP_EDGE,
    connectionstyle="arc3,rad=-0.55",
    linestyle="--",
)
ax.add_patch(fb)
ax.text(5.6, 4.55,
        "append y_hat(t+1)\nshift lookback\n& calendar windows",
        ha="center", va="center", fontsize=8.5,
        color=LOOP_EDGE, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor=LOOP_EDGE,
                  boxstyle="round,pad=0.25"))


# === Legend =============================================================
legend_patches = [
    mpatches.Patch(facecolor=COL_INPUT,  edgecolor=EDGE, label="Input"),
    mpatches.Patch(facecolor=COL_GAT,    edgecolor=EDGE, label="GATConv layer"),
    mpatches.Patch(facecolor=COL_CONCAT, edgecolor=EDGE, label="Concatenate"),
    mpatches.Patch(facecolor=COL_MLP,    edgecolor=EDGE, label="MLP block"),
    mpatches.Patch(facecolor=COL_OUTPUT, edgecolor=EDGE, label="1-step output"),
    mpatches.Patch(facecolor=COL_LOOP,   edgecolor=LOOP_EDGE,
                   label="Recursive loop"),
]
ax.legend(handles=legend_patches, loc="lower left",
          bbox_to_anchor=(0.005, 0.005), frameon=True, fontsize=8, ncol=6)


# ----- Save ---------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(out_dir, "gat_mlp_architecture.png")
pdf_path = os.path.join(out_dir, "gat_mlp_architecture.pdf")
plt.tight_layout()
plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
