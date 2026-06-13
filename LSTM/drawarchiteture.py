"""
Draw the architecture of the LSTM baseline (local, autoregressive LSTM with
exogenous inputs).

Pipeline (matches LSTMBaseline/lstm.py + the recursive inference routine):

    input window  (B, L, 1 + p)   [ y_t  ||  exog_t ]  per step
        -> LSTM (hidden = 32, layers = 1, batch_first)
        -> take LAST hidden state  (B, hidden)
        -> Dropout -> Linear(hidden -> 1)
        -> one-step forecast  y_hat(t+1)
        -> recursive roll-out over the test horizon

Run:
    python drawarchiteture.py
Outputs ``lstm_baseline_architecture.png`` (and .pdf) next to this script.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── palette ────────────────────────────────────────────────────────────────
C_INPUT = "#AED6F1"   # inputs
C_SEQ   = "#D6EAF8"   # per-step sequence
C_LSTM  = "#C8A2DA"   # LSTM
C_PICK  = "#D5DBDB"   # last-state selection
C_HEAD  = "#FAD7A0"   # dropout + linear head
C_OUT   = "#A9DFBF"   # forecast
C_LOSS  = "#F1948A"   # loss
C_EDGE  = "#555555"


def draw_lstm_baseline(
    seq_len: int = 30,
    n_exog: int = 28,
    hidden: int = 32,
    num_layers: int = 1,
    save_path: str = None,
):
    in_width = 1 + n_exog

    fig, ax = plt.subplots(figsize=(11, 12.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(-1.0, 15.5)
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
            ax.text(cx, cy - 0.20, sub, ha="center", va="center",
                    fontsize=sub_fs, color="#333")

    def varrow(x, y0, y1, color=C_EDGE, lw=1.5, text=None, dx=0.28):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                     mutation_scale=14, lw=lw, color=color))
        if text:
            ax.text(x + dx, (y0 + y1) / 2, text, ha="left", va="center",
                    fontsize=7.5, color=color, style="italic")

    cx = 5.5
    ax.text(cx, 15.1, "LSTM Baseline  (local autoregressive LSTM with exogenous inputs)",
            ha="center", fontsize=13, fontweight="bold")

    # ── (1) inputs ─────────────────────────────────────────────────────────
    y_in = 13.8
    box(2.9, y_in, 3.6, 0.85, "Target series  (MinMax scaled)", C_INPUT, 9,
        sub="$y_{t-L} \\,\\dots\\, y_{t-1}$")
    box(7.7, y_in, 4.0, 0.85, f"Exogenous features ({n_exog} cols)", C_INPUT, 9,
        sub="calendar / holiday (known a priori)")

    # ── (2) per-step concatenated sequence ─────────────────────────────────
    y_seq = 12.2
    varrow(2.9, y_in - 0.45, y_seq + 0.55)
    varrow(7.7, y_in - 0.45, y_seq + 0.55)
    box(cx, y_seq, 8.4, 0.95, "Input window", C_SEQ, 9.5, bold=True,
        sub=f"[ $y_\\tau \\;\\Vert\\;$ exog$_\\tau$ ]  →  (B, L = {seq_len}, 1 + {n_exog} = {in_width})")

    # ── (3) unrolled LSTM cells ────────────────────────────────────────────
    y_cells = 10.2
    varrow(cx, y_seq - 0.5, y_cells + 0.95)
    n_show = 5
    xs = [2.2, 3.7, 5.2, 7.4, 8.9]
    labels = ["$x_1$", "$x_2$", "$x_3$", "$x_{L-1}$", "$x_L$"]
    cellw = 0.95
    for i, (xc, lb) in enumerate(zip(xs, labels)):
        ec = "#7D3C98" if i == n_show - 1 else C_EDGE
        lw = 2.0 if i == n_show - 1 else 1.3
        ax.add_patch(FancyBboxPatch((xc - cellw / 2, y_cells - 0.5), cellw, 1.0,
                     boxstyle="round,pad=0.03", linewidth=lw, edgecolor=ec,
                     facecolor=C_LSTM))
        ax.text(xc, y_cells + 0.12, "LSTM", ha="center", va="center",
                fontsize=8, fontweight="bold")
        ax.text(xc, y_cells - 0.22, lb, ha="center", va="center", fontsize=7.5)
        ax.add_patch(FancyArrowPatch((xc, y_cells - 0.85), (xc, y_cells - 0.5),
                     arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#888"))
        ax.text(xc, y_cells - 1.02, lb, ha="center", va="center", fontsize=6.5,
                color="#888")
    for a, b in zip(xs[:n_show - 1], xs[1:n_show]):
        if b - a < 1.6:
            ax.add_patch(FancyArrowPatch((a + cellw / 2, y_cells),
                         (b - cellw / 2, y_cells), arrowstyle="-|>",
                         mutation_scale=10, lw=1.2, color=C_EDGE))
    ax.text(6.3, y_cells, "$\\cdots$", ha="center", va="center", fontsize=14)
    ax.text(cx, y_cells + 1.25, f"shared LSTM unrolled over L = {seq_len} steps "
            f"(hidden = {hidden}, layers = {num_layers}, batch\\_first)",
            ha="center", fontsize=8, style="italic", color="#444")

    # ── (4) last hidden state ──────────────────────────────────────────────
    y_last = 8.3
    ax.add_patch(FancyArrowPatch((8.9, y_cells - 0.5), (cx, y_last + 0.45),
                 arrowstyle="-|>", mutation_scale=13, lw=1.6, color="#7D3C98",
                 connectionstyle="arc3,rad=-0.2"))
    ax.text(7.7, 8.95, "keep last\nhidden state $h_L$", ha="center", va="center",
            fontsize=7.5, color="#7D3C98", style="italic")
    box(cx, y_last, 4.6, 0.8, "Last hidden state", C_PICK, 9.5, bold=True,
        sub=f"$h_L \\in \\mathbb{{R}}^{{{hidden}}}$  (B, {hidden})")

    # ── (5) head ───────────────────────────────────────────────────────────
    y_head = 6.9
    varrow(cx, y_last - 0.4, y_head + 0.4)
    box(cx, y_head, 5.6, 0.8, f"Dropout → Linear({hidden} → 1)", C_HEAD, 9.5, bold=True)

    # ── (6) forecast ───────────────────────────────────────────────────────
    y_out = 5.5
    varrow(cx, y_head - 0.4, y_out + 0.4)
    box(cx, y_out, 3.4, 0.8, "Forecast  $\\hat{y}_{t}$", C_OUT, 10, bold=True,
        sub="(inverse-scaled)")

    # ── (7) loss ───────────────────────────────────────────────────────────
    y_loss = 4.1
    varrow(cx, y_out - 0.4, y_loss + 0.4)
    box(cx, y_loss, 4.4, 0.75, "MSE Loss   $\\mathcal{L} = \\|\\hat{y}-y\\|^2$",
        C_LOSS, 9, bold=True)

    # backprop (single optimiser over all params)
    bp_x = 9.6
    ax.add_patch(FancyArrowPatch((bp_x, y_loss), (bp_x, y_cells + 0.5),
                 arrowstyle="-|>", mutation_scale=15, lw=1.8, color="#C0392B"))
    ax.text(bp_x + 0.22, (y_cells + y_loss) / 2,
            "AdamW\n$\\partial\\mathcal{L}/\\partial\\theta$", ha="center",
            va="center", fontsize=8, color="#C0392B", fontweight="bold")

    # recursive roll-out (feed y_hat back into the window)
    ax.add_patch(FancyArrowPatch((cx - 1.7, y_out), (2.9, y_seq - 0.45),
                 arrowstyle="-|>", mutation_scale=14, lw=1.6, color="#1F618D",
                 connectionstyle="arc3,rad=0.35", linestyle=(0, (5, 3))))
    ax.text(1.45, 8.7, "recursive roll-out:\nappend $\\hat{y}$, drop oldest,\n"
            "advance exog by calendar",
            ha="center", va="center", fontsize=7.5, color="#1F618D", style="italic")

    # ── legend ─────────────────────────────────────────────────────────────
    legend = [
        mpatches.Patch(color=C_INPUT, label="Inputs"),
        mpatches.Patch(color=C_LSTM,  label="LSTM (shared cell)"),
        mpatches.Patch(color=C_HEAD,  label="Dropout + Linear head"),
        mpatches.Patch(color=C_OUT,   label="Forecast"),
        mpatches.Patch(color=C_LOSS,  label="Loss"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.04),
              ncol=5, fontsize=8, frameon=True, edgecolor="#ccc")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=170, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"Saved to {save_path} (+ .pdf)")
    else:
        plt.show()


if __name__ == "__main__":
    # Defaults match the LSTM baseline config: L=30, p=28 (lags off),
    # hidden=32, 1 layer.
    draw_lstm_baseline(
        seq_len=30, n_exog=28, hidden=32, num_layers=1,
        save_path=os.path.join(SCRIPT_DIR, "lstm_baseline_architecture.png"),
    )
