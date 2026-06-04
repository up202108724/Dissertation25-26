import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_lstm_architecture(
    seq_len: int = 7,
    input_size: int = 32,
    n_exog: int = 31,
    hidden_size: int = 32,
    save_path: str = None,
):
    """
    Draws the local LSTM forecaster architecture as a vertical flow diagram.

    Parameters
    ----------
    seq_len     : lookback window length (default 7)
    input_size  : total features per step = 1 + n_exog (default 32)
    n_exog      : number of exogenous columns (default 31)
    hidden_size : LSTM hidden size (default 32)
    save_path   : if given, saves the figure to this path; otherwise shows it
    """

    fig, ax = plt.subplots(figsize=(7, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ------------------------------------------------------------------
    # Colour palette
    # ------------------------------------------------------------------
    C_INPUT   = '#AED6F1'   # light blue  – raw inputs
    C_CONCAT  = '#A9DFBF'   # light green – concatenation / shape note
    C_LSTM    = '#C39BD3'   # purple      – LSTM core
    C_HEAD    = '#FAD7A0'   # pale orange – output head
    C_OUTPUT  = '#A9DFBF'   # light green – forecast output
    C_EDGE    = '#555555'

    def box(ax, cx, cy, w, h, label, sublabel, color, fontsize=9):
        """Draw a rounded rectangle with a bold label and an optional sublabel."""
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor=C_EDGE,
            facecolor=color, zorder=3
        )
        ax.add_patch(rect)
        ax.text(cx, cy + (0.12 if sublabel else 0), label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', zorder=4)
        if sublabel:
            ax.text(cx, cy - 0.25, sublabel,
                    ha='center', va='center', fontsize=7.5,
                    color='#333333', zorder=4)

    def arrow(ax, x, y_start, y_end):
        ax.annotate('', xy=(x, y_end + 0.02),
                    xytext=(x, y_start - 0.02),
                    arrowprops=dict(arrowstyle='->', color=C_EDGE, lw=1.4),
                    zorder=5)

    # ------------------------------------------------------------------
    # Layout (y positions, top → bottom)
    # ------------------------------------------------------------------
    cx = 5.0   # horizontal centre for the main column

    # --- Input boxes (side by side) ---
    y_inp = 12.8
    box(ax, 3.2, y_inp, 3.2, 0.9,
        'Target Time Series', '(MinMax Scaled)', C_INPUT)
    box(ax, 7.3, y_inp, 3.5, 0.9,
        'Exogenous Features',
        f'Calendar, Holidays, Lags — {n_exog} cols', C_INPUT)

    # Converging arrows from both inputs down to concat
    y_concat = 11.4
    ax.annotate('', xy=(cx, y_concat + 0.4),
                xytext=(3.2, y_inp - 0.45),
                arrowprops=dict(arrowstyle='->', color=C_EDGE, lw=1.2), zorder=5)
    ax.annotate('', xy=(cx, y_concat + 0.4),
                xytext=(7.3, y_inp - 0.45),
                arrowprops=dict(arrowstyle='->', color=C_EDGE, lw=1.2), zorder=5)

    # --- Feature Concatenation ---
    box(ax, cx, y_concat, 6.5, 0.9,
        'Feature Concatenation',
        f'[ Target  |  Exog. Features ]   —   input shape per step: ({seq_len}, {input_size})',
        C_CONCAT)

    # --- LSTM ---
    y_lstm = 9.9
    arrow(ax, cx, y_concat, y_lstm)
    box(ax, cx, y_lstm, 6.5, 0.9,
        f'LSTM  (hidden={hidden_size},  num_layers=1,  batch_first=True)',
        None, C_LSTM, fontsize=9)

    # --- Last hidden state note ---
    y_lhs = 8.9
    ax.annotate('', xy=(cx, y_lhs + 0.05),
                xytext=(cx, y_lstm - 0.45),
                arrowprops=dict(arrowstyle='->', color=C_EDGE, lw=1.2), zorder=5)
    ax.text(cx, y_lhs, '[ last hidden state ]',
            ha='center', va='center', fontsize=8,
            color='#555555', style='italic', zorder=4)

    # --- Output head ---
    y_head = 7.9
    arrow(ax, cx, y_lhs, y_head)
    box(ax, cx, y_head, 6.5, 0.9,
        f'Dropout  →  Linear({hidden_size} → 1)',
        None, C_HEAD, fontsize=9)

    # --- Output ---
    y_out = 6.6
    arrow(ax, cx, y_head, y_out)
    box(ax, cx, y_out, 3.5, 0.8,
        'Forecast  t + 1', '(inverse-scaled)', C_OUTPUT, fontsize=10)

    # ------------------------------------------------------------------
    # Title & legend
    # ------------------------------------------------------------------
    ax.set_title('LSTM  Architecture', fontsize=13, fontweight='bold', pad=12)

    legend_items = [
        mpatches.Patch(color=C_INPUT,  label='Raw Inputs'),
        mpatches.Patch(color=C_CONCAT, label='Concatenation / Output'),
        mpatches.Patch(color=C_LSTM,   label='LSTM Core'),
        mpatches.Patch(color=C_HEAD,   label='Output Head'),
    ]
    ax.legend(handles=legend_items, loc='lower center',
              bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=8,
              frameon=True, edgecolor='#cccccc')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    draw_lstm_architecture(save_path='lstm_architecture.png')
