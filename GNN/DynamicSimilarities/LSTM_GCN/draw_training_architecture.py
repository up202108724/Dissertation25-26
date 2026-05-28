"""
Training-architecture diagram for the GCN + LSTM forecaster.

Whereas ``draw_architecture.py`` shows the *model* (data → forecast) graph,
this diagram emphasises the **training loop**: how each mini-batch flows
through the dataset → model → loss → optimiser → parameters of *both* the
GCN and the LSTM (joint end-to-end training).

Run:
    python draw_training_architecture.py
Produces:
    gcn_lstm_training_architecture.png
    gcn_lstm_training_architecture.pdf
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


COLORS = {
    "data":    "#DCE9F7",   # light blue   - data sources
    "loader":  "#D6ECD2",   # light green  - dataset / loader
    "graph":   "#FFE6CC",   # light orange - per-batch graph
    "gcn":     "#F4CCCC",   # light red    - GCN encoder
    "embed":   "#FFF2CC",   # light yellow - z embedding
    "concat":  "#E0CFEE",   # light purple - feature fusion
    "lstm":    "#C8A2DA",   # purple       - LSTM
    "head":    "#FCE4B6",   # peach        - head
    "loss":    "#F8CBAD",   # salmon       - loss
    "optim":   "#B6D7A8",   # green        - optimiser
}
EDGE = "#444444"
GRAD = "#2E7D32"      # dashed green for backward pass


def box(ax, xy, w, h, text, color, fontsize=9, bold=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.2, edgecolor=EDGE, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        wrap=True,
    )
    return {
        "left":   (x,         y + h / 2),
        "right":  (x + w,     y + h / 2),
        "top":    (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
        "cx":     x + w / 2,
        "cy":     y + h / 2,
    }


def arrow(ax, p1, p2, style="-", curve=0.0, color=EDGE, lw=1.2, label=None):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2,
            arrowstyle="-|>", mutation_scale=14,
            linewidth=lw, color=color,
            connectionstyle=f"arc3,rad={curve}",
            linestyle=style,
        )
    )
    if label:
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        ax.text(mx, my, label, fontsize=8, color=color,
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


def main():
    fig, ax = plt.subplots(figsize=(14, 9.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 10.5)
    ax.axis("off")

    # Title
    ax.text(7.0, 10.05,
            "GCN + LSTM — Training Architecture (jointly trained)",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # ── data sources (top row) ──────────────────────────────────────────────
    d_ts   = box(ax, (0.4, 8.6),  3.0, 0.9,
                 "Target + Exog\n(scaled, T × (1 + n_exog))", COLORS["data"])
    d_nx   = box(ax, (4.0, 8.6),  3.6, 0.9,
                 "Per-window Similarity Graphs\n"
                 "(NetworkX list, one per sliding window)", COLORS["data"])
    d_wv   = box(ax, (8.2, 8.6),  3.4, 0.9,
                 "Raw window values\n(per node × window_size)", COLORS["data"])
    d_seed = box(ax, (12.0, 8.6), 1.7, 0.9,
                 "Seed\n+ Splits", COLORS["data"])

    # ── dataset + collate (light green band) ────────────────────────────────
    ds = box(ax, (0.4, 7.0), 11.2, 1.0,
             "GCNTimeSeriesDataset  +  collate_pyg_ts(...)\n"
             "yields (pyg_batch, ts_batch, y_batch, target_idx)",
             COLORS["loader"], bold=True)
    for src in [d_ts, d_nx, d_wv]:
        arrow(ax, src["bottom"], (src["cx"], ds["top"][1]))
    arrow(ax, d_seed["bottom"], (12.0, ds["top"][1]), curve=-0.2)

    # ── per-batch graph block ───────────────────────────────────────────────
    g_batch = box(ax, (0.4, 5.4), 3.6, 1.0,
                  "pyg_batch\n(Batch of B ego-graphs,\nedge_index, edge_attr, x)",
                  COLORS["graph"])
    ts_batch = box(ax, (4.4, 5.4), 3.2, 1.0,
                   "ts_batch\n(B, L, 1 + n_exog)", COLORS["graph"])
    y_batch  = box(ax, (8.0, 5.4), 1.8, 1.0,
                   "y_batch\n(B, 1)", COLORS["graph"])
    tgt_idx  = box(ax, (10.2, 5.4), 1.4, 1.0,
                   "target_idx\n= ptr[:-1]", COLORS["graph"])

    # arrows ds -> per-batch
    for blk, dx in [(g_batch, 2.2), (ts_batch, 6.0),
                    (y_batch, 8.9), (tgt_idx, 10.9)]:
        arrow(ax, (dx, ds["bottom"][1]), (blk["cx"], blk["top"][1]))

    # ── model pipeline (two columns merging into LSTM) ──────────────────────
    gcn = box(ax, (0.4, 3.8), 3.6, 1.0,
              "GCN Encoder\nGCNConv → ReLU → Dropout → GCNConv",
              COLORS["gcn"], bold=True)
    z   = box(ax, (0.4, 2.6), 3.6, 0.7,
              "z = LayerNorm(h[target_idx])  →  (B, d_g=16)",
              COLORS["embed"])

    concat = box(ax, (4.4, 2.6), 5.4, 1.9,
                 "Broadcast z over L and Concatenate\n"
                 "[ ts_batch  ‖  z_seq ]\nper-step input: (B, L, 1 + 28 + 16 = 45)",
                 COLORS["concat"])

    lstm = box(ax, (4.4, 0.9), 5.4, 0.9,
               "LSTM (hidden=32, num_layers=1)  →  Dropout  →  Linear(→1)",
               COLORS["lstm"], bold=True)

    pred = box(ax, (10.4, 0.9), 1.6, 0.9, "ŷ\n(B, 1)", COLORS["head"], bold=True)

    # forward arrows
    arrow(ax, g_batch["bottom"], gcn["top"])
    arrow(ax, gcn["bottom"], z["top"])
    arrow(ax, z["right"], (concat["left"][0], 2.95), curve=-0.05, label="z broadcast")
    arrow(ax, ts_batch["bottom"], (concat["cx"], concat["top"][1]))
    arrow(ax, concat["bottom"], lstm["top"])
    arrow(ax, lstm["right"], pred["left"])

    # ── loss & optimiser (right column) ─────────────────────────────────────
    loss = box(ax, (12.2, 0.9), 1.6, 0.9,
               "MSE / MAE /\nHuber loss", COLORS["loss"], bold=True)
    arrow(ax, pred["right"], loss["left"])
    arrow(ax, (y_batch["cx"], y_batch["bottom"][1]),
          (loss["cx"], loss["top"][1]),
          curve=0.25, label="targets")

    optim = box(ax, (12.2, 2.6), 1.6, 0.9,
                "AdamW\n+ grad-clip 1.0", COLORS["optim"], bold=True)
    sched = box(ax, (12.2, 3.8), 1.6, 0.9,
                "ReduceLROnPlateau\n+ EarlyStopping", COLORS["optim"])

    # ── backward pass (dashed green) — touches *all* trainables ─────────────
    # loss -> head -> LSTM -> concat -> GCN -> params
    arrow(ax, loss["top"], (lstm["cx"] + 1.0, lstm["top"][1] + 0.05),
          style=(0, (4, 3)), curve=-0.25, color=GRAD,
          label="∂L/∂θ_LSTM")
    arrow(ax, (lstm["cx"] - 1.5, lstm["top"][1] + 0.05),
          (concat["cx"], concat["bottom"][1]),
          style=(0, (4, 3)), curve=0.0, color=GRAD)
    arrow(ax, (concat["left"][0], concat["cy"]),
          (z["right"][0] + 0.0, z["right"][1]),
          style=(0, (4, 3)), curve=0.15, color=GRAD)
    arrow(ax, z["bottom"], (gcn["cx"] + 0.6, gcn["bottom"][1] - 0.0),
          style=(0, (4, 3)), curve=0.2, color=GRAD,
          label="∂L/∂θ_GCN")

    # optimiser update arrows (dashed green) onto GCN and LSTM
    arrow(ax, optim["left"], (lstm["right"][0] - 0.1, lstm["right"][1] + 0.15),
          style=(0, (2, 2)), curve=0.0, color=GRAD,
          label="params update")
    arrow(ax, optim["left"], (gcn["right"][0] + 0.1, gcn["right"][1]),
          style=(0, (2, 2)), curve=-0.45, color=GRAD)

    # scheduler -> optimiser
    arrow(ax, sched["bottom"], optim["top"], color=EDGE, style="-", curve=0.0,
          label="val_loss")

    # ── legend ──────────────────────────────────────────────────────────────
    leg_x, leg_y = 0.4, -0.25
    ax.add_patch(Rectangle((leg_x, leg_y), 6.6, 0.55,
                           facecolor="#FAFAFA", edgecolor="#CCCCCC", lw=0.8))
    ax.plot([leg_x + 0.2, leg_x + 0.7], [leg_y + 0.30, leg_y + 0.30],
            color=EDGE, lw=1.4)
    ax.text(leg_x + 0.85, leg_y + 0.30, "forward pass",
            fontsize=8, va="center")
    ax.plot([leg_x + 2.5, leg_x + 3.0], [leg_y + 0.30, leg_y + 0.30],
            color=GRAD, lw=1.4, linestyle=(0, (4, 3)))
    ax.text(leg_x + 3.15, leg_y + 0.30, "backward / gradient",
            fontsize=8, va="center", color=GRAD)
    ax.plot([leg_x + 5.0, leg_x + 5.5], [leg_y + 0.30, leg_y + 0.30],
            color=GRAD, lw=1.4, linestyle=(0, (2, 2)))
    ax.text(leg_x + 5.65, leg_y + 0.30, "optimiser update",
            fontsize=8, va="center", color=GRAD)

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(out_dir, "gcn_lstm_training_architecture.png")
    pdf = os.path.join(out_dir, "gcn_lstm_training_architecture.pdf")
    plt.savefig(png, dpi=220, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
