"""
draw_architecture.py
--------------------------------------------------------------------------
Block-diagram generator for the four end-to-end graph-hybrid forecasters
of the assemble:

        GCN-LSTM   GCN-MLP   GAT-LSTM   GAT-MLP

Each figure mirrors the architecture described in Chapter 8 (Pipelines):
per-step ego-graphs -> shared GNN encoder (GCN or GAT) -> per-step
embedding z -> fusion with [y || exog] -> temporal head (LSTM or
time-distributed MLP) -> one-step forecast -> MSE loss, with the joint
backprop path (GNN + temporal updated together) and the leak-safe
recursive roll-out.

Pure matplotlib -- no graphviz / external deps.

Usage:
    python draw_architecture.py
Outputs PDF + PNG for each model in ./architecture_figures/

Key dimensions are arguments to draw() so you can match your exact run:
    node_feat_dim : 27  (catch24 + min/max/last)   <- the corrected value
    gnn_hidden    : 64
    gnn_out       : 16
    gat_heads     : 4
    n_exog        : 28 for LSTM family, 26 for MLP family (set per call)
    lstm_hidden   : 32
    mlp_hidden    : (128, 64)
    L             : 30  (lookback)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# --------------------------------------------------------------------------
# Palette (mirrors the legend colours used in the thesis figures)
# --------------------------------------------------------------------------
C = {
    "input":    "#AED6F1",   # inputs
    "gnn":      "#F9E79F",   # GCN / GAT encoder + embedding
    "fusion":   "#A9DFBF",   # per-step concatenation
    "temporal": "#D2B4DE",   # LSTM / time-distributed MLP
    "pool":     "#FAD7A0",   # temporal pooling
    "head":     "#F8C471",   # linear head / forecast
    "loss":     "#F1948A",   # MSE loss
}
EDGE = "#34495e"
BP_C = "#c0392b"   # backprop colour
RR_C = "#1f618d"   # recursive roll-out colour


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------
def box(ax, cx, cy, w, h, text, color, fs=9, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2.0, cy - h / 2.0), w, h,
        boxstyle="round,pad=0.15,rounding_size=1.2",
        linewidth=1.3, edgecolor=EDGE, facecolor=color, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, zorder=3, fontweight=weight)
    return dict(cx=cx, cy=cy, w=w, h=h)


def pt(b, side):
    cx, cy, w, h = b["cx"], b["cy"], b["w"], b["h"]
    return {
        "top":    (cx, cy + h / 2.0),
        "bottom": (cx, cy - h / 2.0),
        "left":   (cx - w / 2.0, cy),
        "right":  (cx + w / 2.0, cy),
    }[side]


def arrow(ax, pa, pb, ls="-", color="#2c3e50", lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        pa, pb, arrowstyle="-|>", mutation_scale=15,
        color=color, lw=lw, linestyle=ls,
        connectionstyle="arc3,rad=%s" % rad, zorder=1))


def line(ax, pa, pb, ls="--", color="#2c3e50", lw=1.3):
    ax.add_patch(FancyArrowPatch(
        pa, pb, arrowstyle="-", mutation_scale=1,
        color=color, lw=lw, linestyle=ls, zorder=1))


# --------------------------------------------------------------------------
# Encoder text blocks
# --------------------------------------------------------------------------
def encoder_text(gnn, fin, hid, out, heads):
    emb = r"target-node embedding  z $\in \mathbb{R}^{%d}$" % out
    if gnn == "GCN":
        return ("GCN Encoder  (shared over B·L graphs)\n"
                "GCNConv(%d -> %d) · ReLU · Dropout\n"
                "GCNConv(%d -> %d) · LayerNorm\n" % (fin, hid, hid, out)) + emb
    else:
        per = hid // heads
        return ("GAT Encoder  (edge-weighted, shared)\n"
                "GATConv(%d -> %d × %d heads, concat=%d) · ELU\n"
                "GATConv(-> %d, 1 head, avg) · LayerNorm\n"
                % (fin, per, heads, hid, out)) + emb


# --------------------------------------------------------------------------
# Main figure
# --------------------------------------------------------------------------
def draw(gnn="GCN", temporal="LSTM",
         node_feat_dim=27, gnn_hidden=64, gnn_out=16, gat_heads=4,
         n_exog=28, lstm_hidden=32, mlp_hidden=(128, 64), L=30,
         outdir="architecture_figures"):

    fused = gnn_out + 1 + n_exog

    fig, ax = plt.subplots(figsize=(9.2, 11.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 98.2,
            "%s-%s   (per-step ego-graphs, jointly trained)" % (gnn, temporal),
            ha="center", va="center", fontsize=14, fontweight="bold")

    # ---- inputs -----------------------------------------------------------
    g_in = box(ax, 27, 90, 42, 9,
               "Per-step ego-graphs\n%d graphs · %d node feats each\n(catch24 + min/max/last)"
               % (L, node_feat_dim),
               C["input"], fs=8.5)
    t_in = box(ax, 74, 90, 42, 9,
               "Target + Exogenous\n[ y$_t$ || exog$_t$ ]   ·   %d feats/step" % (1 + n_exog),
               C["input"], fs=8.5)

    # ---- GNN encoder ------------------------------------------------------
    enc = box(ax, 27, 72, 44, 13,
              encoder_text(gnn, node_feat_dim, gnn_hidden, gnn_out, gat_heads),
              C["gnn"], fs=7.8)
    arrow(ax, pt(g_in, "bottom"), pt(enc, "top"))

    # ---- per-step embedding ----------------------------------------------
    z = box(ax, 27, 58, 42, 7,
            "z   (B, L=%d, %d)\nreshaped from (B·L, %d)" % (L, gnn_out, gnn_out),
            C["gnn"], fs=8)
    arrow(ax, pt(enc, "bottom"), pt(z, "top"))

    # ---- fusion -----------------------------------------------------------
    fus = box(ax, 50, 46, 62, 8,
              "Per-step concatenation   [ z || y || exog ]\n(B, %d, %d + 1 + %d = %d)"
              % (L, gnn_out, n_exog, fused),
              C["fusion"], fs=8.5)
    arrow(ax, pt(z, "bottom"),    (41, pt(fus, "top")[1] + 0.2), rad=-0.12)
    arrow(ax, pt(t_in, "bottom"), (60, pt(fus, "top")[1] + 0.2), rad=0.12)

    # ---- temporal head ----------------------------------------------------
    if temporal == "LSTM":
        temp = box(ax, 50, 33, 56, 9,
                   "LSTM (hidden=%d, 1 layer, batch_first)\n"
                   "last hidden state -> Dropout -> Linear(-> 1)" % lstm_hidden,
                   C["temporal"], fs=8.5)
        arrow(ax, pt(fus, "bottom"), pt(temp, "top"))
        out = box(ax, 50, 20, 36, 6.5,
                  r"Forecast  $\hat{y}_{t+1}$  (B, 1)   ·  inverse-scaled",
                  C["head"], fs=8.5)
        arrow(ax, pt(temp, "bottom"), pt(out, "top"))
    else:
        h1, h2 = mlp_hidden
        tdmlp = box(ax, 50, 35, 62, 8,
                    "Time-Distributed MLP  (shared weights over %d steps)\n"
                    "Linear(-> %d)·ReLU·Drop · Linear(-> %d)·ReLU·Drop  ->  (B, %d, %d)"
                    % (L, h1, h2, L, h2),
                    C["temporal"], fs=7.6)
        arrow(ax, pt(fus, "bottom"), pt(tdmlp, "top"))
        pool = box(ax, 50, 24, 54, 7,
                   "Temporal pooling\nconcat[ last (B,%d) || mean (B,%d) ]  ->  (B, %d)"
                   % (h2, h2, 2 * h2),
                   C["pool"], fs=8)
        arrow(ax, pt(tdmlp, "bottom"), pt(pool, "top"))
        out = box(ax, 50, 14, 44, 6.5,
                  r"Linear(%d -> 1)  ->  $\hat{y}_{t+1}$  (B, 1)" % (2 * h2),
                  C["head"], fs=8.5)
        arrow(ax, pt(pool, "bottom"), pt(out, "top"))

    # ---- loss -------------------------------------------------------------
    loss = box(ax, 50, 5, 32, 6,
               r"MSE Loss   $\mathcal{L}=\|\hat{y}-y\|^2$",
               C["loss"], fs=9)
    arrow(ax, pt(out, "bottom"), pt(loss, "top"))

    # ---- joint backprop (dashed, right spine) -----------------------------
    bx = 91.0
    line(ax, pt(loss, "right"), (bx, 5), ls="--", color=BP_C, lw=1.3)
    line(ax, (bx, 5), (bx, 72), ls="--", color=BP_C, lw=1.3)
    arrow(ax, (bx, 72), pt(enc, "right"), ls="--", color=BP_C, lw=1.3, rad=0.0)
    ax.text(bx + 4.5, 40,
            "backprop  " + r"$\partial\mathcal{L}/\partial\theta$" +
            "\n(GNN + temporal\nupdated jointly)",
            rotation=90, ha="center", va="center", fontsize=7.6, color=BP_C)

    # ---- recursive roll-out (dashed, left spine) --------------------------
    rx = 3.5
    oy = pt(out, "left")[1]
    line(ax, pt(out, "left"), (rx, oy), ls="--", color=RR_C, lw=1.3)
    line(ax, (rx, oy), (rx, 90), ls="--", color=RR_C, lw=1.3)
    arrow(ax, (rx, 90), pt(g_in, "left"), ls="--", color=RR_C, lw=1.3)
    ax.text(rx - 1.6, 48,
            "recursive roll-out\nappend " + r"$\hat{y}$" +
            ", rebuild ego-graph\nfrom own predictions",
            rotation=90, ha="center", va="center", fontsize=7.4, color=RR_C)

    # ---- legend -----------------------------------------------------------
    handles = [
        mpatches.Patch(color=C["input"],    label="Inputs"),
        mpatches.Patch(color=C["gnn"],      label="GNN encoder / embedding"),
        mpatches.Patch(color=C["fusion"],   label="Fusion (concat)"),
        mpatches.Patch(color=C["temporal"], label="Temporal head"),
        mpatches.Patch(color=C["pool"],     label="Pooling"),
        mpatches.Patch(color=C["head"],     label="Linear head / forecast"),
        mpatches.Patch(color=C["loss"],     label="Loss"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, -0.055), fontsize=7.8, frameon=False,
              handlelength=1.1, columnspacing=1.2)

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, "arch_%s_%s" % (gnn.lower(), temporal.lower()))
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return stem


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # n_exog differs by family in the baselines (LSTM=28, MLP=26).
    # Adjust to match your actual feature set if needed.
    configs = [
        dict(gnn="GCN", temporal="LSTM", n_exog=28),
        dict(gnn="GCN", temporal="MLP",  n_exog=26),
        dict(gnn="GAT", temporal="LSTM", n_exog=28),
        dict(gnn="GAT", temporal="MLP",  n_exog=26),
    ]
    for cfg in configs:
        stem = draw(node_feat_dim=27, **cfg)
        print("wrote %s.pdf / .png" % stem)
