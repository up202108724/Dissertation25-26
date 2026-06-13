"""
Draw the TRAINING architecture of the PER-STEP GCN + LSTM forecaster
(one ego-graph per lookback day, GCN and LSTM trained jointly).

Mirrors the reference "training architecture" layout but keeps the correct
per-step batch count:  collate_pyg_ts flattens B samples x L lookback days
into  B*L  ego-graphs, the GCN encodes them in one sparse pass, and the
target-node embedding z is reshaped (B*L, d_g) -> (B, L, d_g) before being
concatenated per step with ts_batch.

Forward pass (solid), backward/gradient (dashed) and optimiser update (dotted)
are drawn so the joint end-to-end training is explicit.

Run:
    python draw_architeture.py
Outputs ``gcn_lstm_training_architecture.png`` (and .pdf) next to this script.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── palette ────────────────────────────────────────────────────────────────
C_INPUT = "#cfe3f7"
C_DATA  = "#cfe9cf"
C_TENS  = "#fde3c4"
C_GCN   = "#f6c9c9"
C_Z     = "#fdeec2"
C_FUSE  = "#d9c8ee"
C_LSTM  = "#c9a8e0"
C_OPT   = "#a9d3a0"
C_LOSS  = "#f3b58a"
C_EDGE  = "#3a3a3a"
C_FWD   = "#3a3a3a"
C_BWD   = "#2e8b57"
C_OPTU  = "#2e8b57"


def box(ax, xy, w, h, text, color, fontsize=9, ec=C_EDGE, lw=1.3, bold=False):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw, edgecolor=ec, facecolor=color, mutation_aspect=1,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=5,
            fontweight="bold" if bold else "normal")
    return {
        "c": (x + w / 2, y + h / 2),
        "n": (x + w / 2, y + h), "s": (x + w / 2, y),
        "e": (x + w, y + h / 2), "w": (x, y + h / 2),
        "ne": (x + w, y + h), "nw": (x, y + h),
        "se": (x + w, y), "sw": (x, y),
    }


def arrow(ax, p0, p1, text=None, color=C_FWD, lw=1.5, rad=0.0,
          fontsize=7.5, dx=0.0, dy=0.14, ls="-", scale=13):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=scale, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=3,
        linestyle=ls,
    )
    ax.add_patch(a)
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + dx, my + dy, text, ha="center", va="center",
                fontsize=fontsize, color=color, zorder=6, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(15.5, 10))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(7.75, 9.6, "Per-step GCN + LSTM — Training Architecture (jointly trained)",
            ha="center", fontsize=15, fontweight="bold")

    # ── Row 1: input sources ───────────────────────────────────────────────
    inp_tgt = box(ax, (0.4, 8.2), 3.0, 0.95,
                  "Target + Exog\n(scaled, T x (1 + n_exog))", C_INPUT, 8.5)
    inp_gph = box(ax, (3.8, 8.2), 3.4, 0.95,
                  "Per-window similarity graphs\n(NetworkX list, 1 per sliding window)",
                  C_INPUT, 8.5)
    inp_raw = box(ax, (7.6, 8.2), 3.2, 0.95,
                  "Raw window values\n(per node x window_size)", C_INPUT, 8.5)
    inp_seed = box(ax, (11.2, 8.2), 2.0, 0.95,
                   "Seed\n+ Splits", C_INPUT, 8.5)

    # ── Row 2: dataset + collate ────────────────────────────────────────────
    data = box(ax, (0.4, 6.85), 12.8, 0.9,
               "GCNTimeSeriesDataset  +  collate_pyg_ts(...)\n"
               "yields (pyg_batch, ts_batch, y_batch, target_idx, L)",
               C_DATA, 9.5, bold=True)
    for src in (inp_tgt, inp_gph, inp_raw):
        arrow(ax, src["s"], (src["s"][0], 7.75))
    arrow(ax, inp_seed["s"], (12.6, 7.75), rad=-0.2)

    # ── Row 3: the batch tensors ────────────────────────────────────────────
    pyg = box(ax, (0.4, 5.35), 3.2, 1.0,
              "pyg_batch\n(B·L ego-graphs flattened,\nedge_index, edge_attr, x)",
              C_TENS, 8)
    tsb = box(ax, (4.1, 5.35), 3.0, 1.0,
              "ts_batch\n(B, L, 1 + n_exog)", C_TENS, 8.5)
    yb  = box(ax, (7.6, 5.35), 2.0, 1.0,
              "y_batch\n(B, 1)", C_TENS, 8.5)
    tidx = box(ax, (10.1, 5.35), 2.4, 1.0,
               "target_idx\n= ptr[:-1]  (B·L,)", C_TENS, 8.5)
    for t in (pyg, tsb, yb, tidx):
        arrow(ax, (t["n"][0], 6.85), t["n"])

    ax.text(2.0, 5.2, "B·L = batch × lookback graphs", ha="center",
            fontsize=7, color="#666", style="italic")

    # ── Row 4: GCN encoder + z ──────────────────────────────────────────────
    gcn = box(ax, (0.4, 3.7), 3.4, 0.95,
              "GCN Encoder\nGCNConv → ReLU → Dropout → GCNConv",
              C_GCN, 8.5, bold=True)
    znorm = box(ax, (0.4, 2.45), 3.4, 0.85,
                "z = LayerNorm(h[target_idx])\n(B·L, d_g=16) → view(B, L, d_g)",
                C_Z, 8)
    arrow(ax, pyg["s"], gcn["n"])
    arrow(ax, tidx["s"], (3.3, 4.65), rad=0.25, dy=0.0)  # target_idx -> gcn select
    arrow(ax, gcn["s"], znorm["n"])

    # ── Fuse: broadcast + concat ────────────────────────────────────────────
    fuse = box(ax, (4.3, 2.55), 5.0, 2.0,
               "Broadcast z over L and concatenate\n[ ts_batch  ‖  z_seq ]\n"
               "per-step input: (B, L, 1 + n_exog + d_g)",
               C_FUSE, 9)
    arrow(ax, tsb["s"], (5.6, 4.55), rad=0.0)
    arrow(ax, znorm["e"], fuse["w"], "z broadcast", color=C_FWD, dy=0.18)

    # ── LSTM + head + prediction ────────────────────────────────────────────
    lstm = box(ax, (4.3, 0.95), 5.6, 0.95,
               "LSTM (hidden=32, num_layers=1) → Dropout → Linear(→1)",
               C_LSTM, 9, bold=True)
    arrow(ax, fuse["s"], (6.8, 1.9))
    pred = box(ax, (10.3, 0.95), 1.7, 0.95, "ŷ\n(B, 1)", C_Z, 10, bold=True)
    arrow(ax, lstm["e"], pred["w"])

    # ── Loss ────────────────────────────────────────────────────────────────
    loss = box(ax, (12.4, 0.95), 2.6, 0.95,
               "MSE loss", C_LOSS, 9.5, bold=True)
    arrow(ax, pred["e"], loss["w"])
    arrow(ax, yb["s"], loss["n"], "targets", rad=-0.35, dy=0.0, dx=0.4)

    # ── Optimiser column ────────────────────────────────────────────────────
    opt = box(ax, (12.7, 4.55), 2.5, 0.9,
              "AdamW\n+ grad-clip 1.0", C_OPT, 9, bold=True)
    sched = box(ax, (12.7, 5.6), 2.5, 0.85,
                "ReduceLROnPlateau\n+ EarlyStopping", C_OPT, 8.5)
    arrow(ax, loss["n"], opt["s"], rad=-0.2)
    arrow(ax, sched["s"], opt["n"], "val_loss", dy=0.12)

    # ── Backward / gradient (dashed) ────────────────────────────────────────
    arrow(ax, loss["w"], pred["e"], color=C_BWD, ls="--", rad=0.0, scale=11,
          dy=-0.2)
    arrow(ax, (6.8, 0.95), fuse["s"], color=C_BWD, ls="--", rad=0.0, scale=11,
          text="∂L/∂θ_LSTM", dx=2.6, dy=-0.05)
    arrow(ax, fuse["w"], znorm["e"], color=C_BWD, ls="--", rad=-0.15, scale=11)
    arrow(ax, znorm["n"], gcn["s"], color=C_BWD, ls="--", rad=0.25, scale=11,
          text="∂L/∂θ_GCN", dx=-0.55, dy=0.0)

    # ── Optimiser update (dotted) back to params ────────────────────────────
    arrow(ax, opt["w"], lstm["e"], color=C_OPTU, ls=":", rad=0.25, scale=11,
          text="params update", dx=-0.2, dy=0.35)
    arrow(ax, opt["w"], gcn["e"], color=C_OPTU, ls=":", rad=0.4, scale=11)

    # ── Legend ──────────────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color=C_FWD, lw=1.6, label="forward pass"),
        plt.Line2D([0], [0], color=C_BWD, lw=1.6, ls="--",
                   label="backward / gradient"),
        plt.Line2D([0], [0], color=C_OPTU, lw=1.6, ls=":",
                   label="optimiser update"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=True,
              ncol=3, bbox_to_anchor=(0.02, 0.0))

    fig.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, "gcn_lstm_training_architecture.png")
    out_pdf = os.path.join(SCRIPT_DIR, "gcn_lstm_training_architecture.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()
