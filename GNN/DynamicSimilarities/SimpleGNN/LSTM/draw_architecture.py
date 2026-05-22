"""
Architecture diagram for SimpleGNNLSTMForecaster
  – GCN encoder (ONE ego-graph per sample)
  – z projects to LSTM h0/c0
  – LSTM processes ts_seq [ value_t | cal_{t+1} ]
Run:  python draw_architecture.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Palette ──────────────────────────────────────────────────────────────────
C_IN    = '#AED6F1'   # blue   – sequence inputs
C_FEAT  = '#FDEBD0'   # peach  – node-feature construction
C_GRAPH = '#A9DFBF'   # green  – graph / GCN
C_PROJ  = '#FAD7A0'   # orange – h0 / c0 projections
C_LSTM  = '#D7BDE2'   # purple – LSTM
C_HEAD  = '#F1948A'   # red    – linear head
ARROW   = '#1A252F'
EDGE    = '#2C3E50'

fig, ax = plt.subplots(figsize=(22, 13))
ax.set_xlim(0, 22)
ax.set_ylim(0, 13)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(x, y, w, h, color, lines, fs=8.5, bold_first=False):
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor=EDGE, linewidth=1.6, zorder=3))
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    for i, line in enumerate(lines):
        dy = (i - (n - 1) / 2) * (fs + 1.5) * 0.014
        ax.text(x, y - dy, line,
                ha='center', va='center', fontsize=fs,
                fontweight='bold' if (bold_first and i == 0) else 'normal',
                zorder=4, color='#1A252F')


def arr(x1, y1, x2, y2, lbl='', sty='arc3,rad=0.0'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=ARROW, lw=1.6,
                                connectionstyle=sty, mutation_scale=14),
                zorder=5)
    if lbl:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2 + 0.1, lbl,
                fontsize=7, color='#555', zorder=6, style='italic')


# ── Column positions ──────────────────────────────────────────────────────────
XA, XB, XC, XD, XE, XF, XG = 1.6, 4.4, 7.6, 10.8, 14.0, 17.5, 21.0

# ── Section banners ───────────────────────────────────────────────────────────
for bx, bw, lbl, col in [
    (XA, 2.6, 'Data Inputs',       '#EBF5FB'),
    (XB, 2.6, 'Node Features',     '#FEF9E7'),
    (XC, 2.8, 'Ego-Graph  (×1)',   '#EAFAF1'),
    (XD, 2.8, 'GCN Encoder',       '#EAFAF1'),
    (XE, 2.8, 'h₀ / c₀  Init',    '#FEF9E7'),
    (XF, 2.8, 'LSTM Forecaster',   '#F5EEF8'),
    (XG, 1.5, 'Output',            '#EAFAF1'),
]:
    ax.add_patch(FancyBboxPatch(
        (bx - bw / 2, 0.3), bw, 12.2,
        boxstyle="round,pad=0.1",
        facecolor=col, edgecolor='#BFC9CA',
        linewidth=1.0, zorder=1, alpha=0.5))
    ax.text(bx, 12.7, lbl,
            ha='center', va='center', fontsize=8.5,
            color='#5D6D7E', fontweight='bold', zorder=4)

# ═══════════════════════════════════════════════════════════════════════════════
#  A – DATA INPUTS
# ═══════════════════════════════════════════════════════════════════════════════
box(XA, 10.5, 2.4, 0.9, '#D5DBDB',
    ['df_wide', '(items × dates)'], fs=8, bold_first=True)
box(XA,  8.8, 2.4, 0.9, C_IN,
    ['Historical Sales (scaled)', 'lookback = 30 days'], fs=8, bold_first=True)
box(XA,  7.1, 2.4, 0.9, C_IN,
    ['Calendar Features (next-step)', 'cal_dim = 31  (EXOG_COLS)'], fs=8, bold_first=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  B – NODE FEATURES
#      feature_dim = lookback(30) + cal_dim(31) + stats_8 = 69
# ═══════════════════════════════════════════════════════════════════════════════
box(XB, 9.85, 2.6, 2.1, C_FEAT,
    ['TARGET NODE',
     '──────────────────',
     'ts_lookback   (30 dims)',
     'cal_next_step (31 dims)',
     'stats_8:  last, mean7, mean,',
     '  std, zero_ratio, slope,',
     '  min, max',
     '──────────────────',
     'feature_dim = 69'],
    fs=7.5, bold_first=True)

box(XB, 7.15, 2.6, 1.6, C_FEAT,
    ['NEIGHBOR NODES',
     '──────────────────',
     'ts_window right-aligned',
     '  in zeros(30)',
     'cal = 0  (unknown)',
     '+ stats_8   (total: 69)'],
    fs=7.5, bold_first=True)

arr(XA + 1.2, 8.8,  XB - 1.3, 9.9,  lbl='lookback ts')
arr(XA + 1.2, 7.1,  XB - 1.3, 9.0,  lbl='cal next-step')
arr(XA + 1.2, 10.5, XB - 1.3, 7.15, lbl='window data')

ax.text(XB, 11.6, 'feature_dim  =  30 + 31 + 8  =  69',
        ha='center', fontsize=8, color='#784212', fontweight='bold', zorder=6)

# ═══════════════════════════════════════════════════════════════════════════════
#  C – EGO-GRAPH  (ONE per sample)
# ═══════════════════════════════════════════════════════════════════════════════
box(XC, 8.6, 2.8, 3.0, C_GRAPH,
    ['Ego-Graph   (ONE per sample)',
     '────────────────────────────',
     'Pairwise similarity (Spearman)',
     'Threshold → edges',
     'Star topology',
     '  + optional within-star links',
     'target_id  →  node index 0',
     'PyG  Data(x, edge_index, edge_attr)'],
    fs=7.5, bold_first=True)

ax.text(XC, 11.6, '1 graph / sample  (not L graphs)',
        ha='center', fontsize=8.2, color='#1A5276', fontweight='bold', zorder=6)

arr(XB + 1.3, 9.85, XC - 1.4, 9.2)
arr(XB + 1.3, 7.15, XC - 1.4, 8.1)

# ═══════════════════════════════════════════════════════════════════════════════
#  D – GCN ENCODER
# ═══════════════════════════════════════════════════════════════════════════════
box(XD, 10.5, 2.6, 0.9, C_GRAPH,
    ['Batch.from_data_list', 'B graphs  →  PyG Batch'],
    fs=8, bold_first=True)

box(XD,  9.1, 2.6, 0.9, C_GRAPH,
    ['GCNConv 1  (add_self_loops=True)',
     'in(69) → hidden(32)',
     'edge_weight = similarity'],
    fs=7.5, bold_first=True)

box(XD,  7.8, 2.6, 0.7, C_GRAPH,
    ['ReLU  +  Dropout(p=0.2)'], fs=8)

box(XD,  6.7, 2.6, 0.9, C_GRAPH,
    ['GCNConv 2  (add_self_loops=True)',
     'hidden(32) → out(16)'],
    fs=7.5, bold_first=True)

box(XD,  5.3, 2.6, 1.1, C_GRAPH,
    ['Extract Target Node',
     'idx  =  pyg_batch.ptr[:-1]',
     '(always node 0 per graph)',
     'z  :  (B, 16)'],
    fs=7.6, bold_first=True)

# Ego-graph → Batch
ax.annotate('', xy=(XD - 1.3, 10.5), xytext=(XC + 1.4, 8.6),
            arrowprops=dict(arrowstyle='-|>', color=ARROW, lw=1.4,
                            connectionstyle='arc3,rad=-0.25'), zorder=5)
arr(XD, 10.05, XD, 9.55)   # Batch → GCNConv1
arr(XD,  9.65, XD, 8.15)   # GCNConv1 → ReLU  (small gap)
arr(XD,  7.45, XD, 7.15)   # ReLU → GCNConv2
arr(XD,  6.25, XD, 5.85)   # GCNConv2 → Extract

# ═══════════════════════════════════════════════════════════════════════════════
#  E – h₀ / c₀ PROJECTIONS
# ═══════════════════════════════════════════════════════════════════════════════
box(XE, 7.5, 2.6, 1.0, C_PROJ,
    ['h₀  projection',
     'Linear(16  →  lstm_hidden × layers)',
     'reshape:  (layers, B, 64)'],
    fs=7.8, bold_first=True)

box(XE, 5.8, 2.6, 1.0, C_PROJ,
    ['c₀  projection',
     'Linear(16  →  lstm_hidden × layers)',
     'reshape:  (layers, B, 64)'],
    fs=7.8, bold_first=True)

arr(XD + 1.3, 5.3, XE - 1.3, 7.5, lbl='z', sty='arc3,rad=-0.25')
arr(XD + 1.3, 5.3, XE - 1.3, 5.8, lbl='z', sty='arc3,rad=0.12')

# ═══════════════════════════════════════════════════════════════════════════════
#  F – LSTM FORECASTER
# ═══════════════════════════════════════════════════════════════════════════════
box(XF, 10.1, 2.6, 0.85, C_IN,
    ['Sequence Input  (per step t)',
     '[ value_t  |  cal_{t+1} ]',
     'ts_seq :  (B, 30, 1+31=32)'],
    fs=7.8, bold_first=True)

box(XF, 7.3, 2.6, 1.5, C_LSTM,
    ['LSTM',
     '─────────────────────',
     'input_size  = 32   (1 + cal_dim)',
     'hidden_size = 64',
     'num_layers  = 1,  batch_first=True',
     '─────────────────────',
     'Initialised with  h₀, c₀',
     'from GCN embedding  z'],
    fs=7.5, bold_first=True)

box(XF, 4.8, 2.6, 0.9, C_HEAD,
    ['Last Hidden  →  Dropout',
     'Linear(64  →  horizon=1)'],
    fs=8, bold_first=True)

# Arrows into LSTM column
arr(XA + 1.2, 8.8,  XF - 1.3, 10.1, lbl='ts_seq', sty='arc3,rad=-0.07')
arr(XF, 9.67, XF, 8.05)                              # ts_seq → LSTM
arr(XE + 1.3, 7.5, XF - 1.3, 7.7, lbl='h₀')
arr(XE + 1.3, 5.8, XF - 1.3, 6.9, lbl='c₀', sty='arc3,rad=-0.12')
arr(XF, 6.55, XF, 5.25)                              # LSTM → head

# ═══════════════════════════════════════════════════════════════════════════════
#  G – OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
box(XG, 4.8, 1.4, 0.85, '#A9DFBF',
    ['Forecast  t+1', '(B, 1, 1)'],
    fs=8.5, bold_first=True)

arr(XF + 1.3, 4.8, XG - 0.7, 4.8)

# ── Recursive-inference note ──────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch(
    (XE - 1.4, 0.3), (XG + 0.75 - XE + 1.4), 1.4,
    boxstyle="round,pad=0.08",
    facecolor='#FDFEFE', edgecolor='#884EA0',
    linewidth=1.3, linestyle=':', zorder=2))
ax.text((XE + XG) / 2, 1.1,
        'Recursive Inference:  forecast(t)  →  appended to ts_lookback  →  ego-graph rebuilt  →  step t+1\n'
        '(152 autoregressive steps to produce the full forecast horizon)',
        ha='center', va='center', fontsize=7.2, color='#6C3483', zorder=6)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(11, 12.3,
        'GCN + LSTM Forecaster  (SimpleGNNLSTMForecaster)  —  Architecture Block Diagram',
        ha='center', va='center', fontsize=12, fontweight='bold',
        color='#1A252F', zorder=6)

plt.tight_layout(pad=0.4)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'architecture_gcn_lstm.png')
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved  →  {out}')
plt.close()
