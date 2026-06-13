"""
Draw the architecture of the TimeDistributed MLP forecaster.

Pipeline (matches TimeDistributedMLP/mlp.py + recursive inference):

    input window  (B, L, C)   [ y_t  ||  exog_t ]  per step
        -> shared MLP applied independently to every timestep
              Linear(C -> 128) -> ReLU -> Dropout
              Linear(128 -> 64) -> ReLU -> Dropout
        -> last step  h[:, -1, :]   (B, 64)   — recency
        -> mean pool  h.mean(dim=1) (B, 64)   — global context
        -> concat  (B, 128)
        -> Linear(128 -> 1)
        -> one-step forecast  y_hat(t+1)
        -> recursive roll-out over the test horizon

Run:
    python draw_architeture.py
Outputs ``tdmlp_architecture.png`` (and .pdf) next to this script.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── palette ────────────────────────────────────────────────────────────────
C_INPUT  = "#AED6F1"   # inputs
C_SEQ    = "#D6EAF8"   # per-step sequence / input window
C_MLP    = "#C8A2DA"   # MLP shared block
C_PICK   = "#D5DBDB"   # aggregation
C_HEAD   = "#FAD7A0"   # output head
C_OUT    = "#A9DFBF"   # forecast
C_LOSS   = "#F1948A"   # loss
C_EDGE   = "#555555"


def draw_tdmlp(
    seq_len: int = 30,
    n_exog: int = 26,
    hidden_sizes=(128, 64),
    dropout: float = 0.2,
    save_path: str = None,
):
    in_channels = 1 + n_exog
    H_last = hidden_sizes[-1]

    fig, ax = plt.subplots(figsize=(11, 13.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(-1.0, 17.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(cx, cy, w, h, label, color, fs=9, bold=False, sub=None, sub_fs=7.5):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.04",
            linewidth=1.3, edgecolor=C_EDGE, facecolor=color))
        yo = 0.16 if sub else 0.0
        ax.text(cx, cy + yo, label, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")
        if sub:
            ax.text(cx, cy - 0.22, sub, ha="center", va="center",
                    fontsize=sub_fs, color="#333")

    def varrow(x, y0, y1, color=C_EDGE, lw=1.5, text=None, dx=0.28):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                     mutation_scale=14, lw=lw, color=color))
        if text:
            ax.text(x + dx, (y0 + y1) / 2, text, ha="left", va="center",
                    fontsize=7.5, color=color, style="italic")

    cx = 5.5

    # ── title ─────────────────────────────────────────────────────────────
    ax.text(cx, 16.5,
            "TimeDistributed MLP  (shared-weight MLP per timestep with exogenous inputs)",
            ha="center", fontsize=12, fontweight="bold")

    # ── (1) inputs ─────────────────────────────────────────────────────────
    y_in = 15.2
    box(2.9, y_in, 3.6, 0.85, "Target series  (MinMax scaled)", C_INPUT, 9,
        sub="$y_{t-L} \\,\\dots\\, y_{t-1}$")
    box(7.7, y_in, 4.0, 0.85, f"Exogenous features ({n_exog} cols)", C_INPUT, 9,
        sub="calendar / holiday (known a priori)")

    # ── (2) input window ───────────────────────────────────────────────────
    y_seq = 13.5
    varrow(2.9, y_in - 0.45, y_seq + 0.55)
    varrow(7.7, y_in - 0.45, y_seq + 0.55)
    box(cx, y_seq, 8.4, 0.95, "Input window", C_SEQ, 9.5, bold=True,
        sub=f"[ $y_\\tau \\;\\Vert\\;$ exog$_\\tau$ ]  →  "
            f"(B, L = {seq_len}, 1 + {n_exog} = {in_channels})")

    # ── (3) shared MLP cells ───────────────────────────────────────────────
    y_cells = 11.3
    varrow(cx, y_seq - 0.5, y_cells + 1.0)

    n_show = 5
    xs = [2.2, 3.7, 5.2, 7.4, 8.9]
    labels_top = ["$x_1$", "$x_2$", "$x_3$", "$x_{L-1}$", "$x_L$"]
    cellw, cellh = 1.05, 1.1
    for i, (xc, lb) in enumerate(zip(xs, labels_top)):
        ec  = "#7D3C98" if i == n_show - 1 else C_EDGE
        lw  = 2.0       if i == n_show - 1 else 1.3
        ax.add_patch(FancyBboxPatch(
            (xc - cellw / 2, y_cells - cellh / 2), cellw, cellh,
            boxstyle="round,pad=0.03", linewidth=lw, edgecolor=ec, facecolor=C_MLP))
        ax.text(xc, y_cells + 0.22, "MLP", ha="center", va="center",
                fontsize=8, fontweight="bold")
        lbl1 = f"$C \\to {hidden_sizes[0]}$"
        lbl2 = f"${hidden_sizes[0]} \\to {H_last}$"
        ax.text(xc, y_cells - 0.05, lbl1, ha="center", va="center",
                fontsize=6.5, color="#333")
        ax.text(xc, y_cells - 0.32, lbl2, ha="center", va="center",
                fontsize=6.5, color="#333")
        # input arrow below
        ax.add_patch(FancyArrowPatch(
            (xc, y_cells - cellh / 2 - 0.35), (xc, y_cells - cellh / 2),
            arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#888"))
        ax.text(xc, y_cells - cellh / 2 - 0.52, lb,
                ha="center", va="center", fontsize=6.5, color="#888")

    # horizontal arrows between adjacent cells (dashed — no recurrence, just visual)
    for a, b in zip(xs[:n_show - 1], xs[1:n_show]):
        if b - a < 1.6:
            ax.add_patch(FancyArrowPatch(
                (a + cellw / 2, y_cells), (b - cellw / 2, y_cells),
                arrowstyle="-", mutation_scale=10, lw=1.0,
                color="#bbb", linestyle=(0, (4, 3))))

    ax.text(6.3, y_cells, "$\\cdots$", ha="center", va="center", fontsize=14)
    ax.text(cx, y_cells + 1.35,
            f"shared MLP applied independently to each of L = {seq_len} timesteps  "
            f"(weights tied across time)",
            ha="center", fontsize=8, style="italic", color="#444")

    # ── (4) aggregation: last + mean ───────────────────────────────────────
    y_agg = 9.15
    # last step arrow (from rightmost cell, highlighted)
    ax.add_patch(FancyArrowPatch(
        (8.9, y_cells - cellh / 2), (6.7, y_agg + 0.42),
        arrowstyle="-|>", mutation_scale=13, lw=1.6, color="#7D3C98",
        connectionstyle="arc3,rad=-0.25"))
    ax.text(8.55, 9.65, "last step\n$h[:,-1,:]$", ha="center", va="center",
            fontsize=7.5, color="#7D3C98", style="italic")
    # mean-pool arrow (from middle, generic)
    ax.add_patch(FancyArrowPatch(
        (cx, y_cells - cellh / 2), (4.5, y_agg + 0.42),
        arrowstyle="-|>", mutation_scale=13, lw=1.6, color="#1A5276",
        connectionstyle="arc3,rad=0.1"))
    ax.text(3.45, 9.65, "mean pool\n$h$.mean(dim=1)", ha="center", va="center",
            fontsize=7.5, color="#1A5276", style="italic")

    box(cx, y_agg, 7.2, 0.85,
        "Aggregate: last step  $\\oplus$  mean pool", C_PICK, 9.5, bold=True,
        sub=f"$(B,\\,{H_last}) \\;\\oplus\\; (B,\\,{H_last})$  →  "
            f"$(B,\\,{H_last * 2})$")

    # ── (5) output head ────────────────────────────────────────────────────
    y_head = 7.65
    varrow(cx, y_agg - 0.45, y_head + 0.4)
    box(cx, y_head, 5.8, 0.8,
        f"Linear({H_last * 2} → 1)", C_HEAD, 9.5, bold=True)

    # ── (6) forecast ───────────────────────────────────────────────────────
    y_out = 6.2
    varrow(cx, y_head - 0.4, y_out + 0.4)
    box(cx, y_out, 3.4, 0.8, "Forecast  $\\hat{y}_{t}$", C_OUT, 10, bold=True,
        sub="(inverse-scaled)")

    # ── (7) loss ───────────────────────────────────────────────────────────
    y_loss = 4.75
    varrow(cx, y_out - 0.4, y_loss + 0.4)
    box(cx, y_loss, 4.4, 0.75,
        "MSE Loss   $\\mathcal{L} = \\|\\hat{y}-y\\|^2$", C_LOSS, 9, bold=True)

    # ── backprop ───────────────────────────────────────────────────────────
    bp_x = 9.6
    ax.add_patch(FancyArrowPatch(
        (bp_x, y_loss), (bp_x, y_cells + 0.5),
        arrowstyle="-|>", mutation_scale=15, lw=1.8, color="#C0392B"))
    ax.text(bp_x + 0.22, (y_cells + y_loss) / 2,
            "AdamW\n$\\partial\\mathcal{L}/\\partial\\theta$",
            ha="center", va="center", fontsize=8,
            color="#C0392B", fontweight="bold")

    # ── recursive roll-out ─────────────────────────────────────────────────
    ax.add_patch(FancyArrowPatch(
        (cx - 1.7, y_out), (2.9, y_seq - 0.48),
        arrowstyle="-|>", mutation_scale=14, lw=1.6, color="#1F618D",
        connectionstyle="arc3,rad=0.35", linestyle=(0, (5, 3))))
    ax.text(1.45, 10.1,
            "recursive roll-out:\nappend $\\hat{y}$, drop oldest,\n"
            "advance exog by calendar",
            ha="center", va="center", fontsize=7.5,
            color="#1F618D", style="italic")

    # ── legend ─────────────────────────────────────────────────────────────
    legend = [
        mpatches.Patch(color=C_INPUT, label="Inputs"),
        mpatches.Patch(color=C_MLP,   label="MLP (shared weights, per timestep)"),
        mpatches.Patch(color=C_PICK,  label="Aggregation (last $\\oplus$ mean)"),
        mpatches.Patch(color=C_HEAD,  label="Linear head"),
        mpatches.Patch(color=C_OUT,   label="Forecast"),
        mpatches.Patch(color=C_LOSS,  label="Loss"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.04),
              ncol=3, fontsize=8, frameon=True, edgecolor="#ccc")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=170, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"Saved to {save_path} (+ .pdf)")
    else:
        plt.show()


if __name__ == "__main__":
    draw_tdmlp(
        seq_len=30,
        n_exog=26,
        hidden_sizes=(128, 64),
        dropout=0.2,
        save_path=os.path.join(SCRIPT_DIR, "tdmlp_architecture.png"),
    )
