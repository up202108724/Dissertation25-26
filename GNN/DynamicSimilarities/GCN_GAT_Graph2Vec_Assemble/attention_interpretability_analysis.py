"""
Interpretability analysis for the GAT assemble — learned attention weights.

Answers the interpretability clause of RQ2: *why* does the GAT branch make the
prediction it does for a target product?  Because ``EdgeWeightedGATConv`` keeps
PyG's learned multi-head attention (then gates it by the similarity edge weight),
we can read, for any ego-graph, exactly how much the target node attends to each
neighbour when it aggregates the graph branch's embedding ``z``.

We snapshot that attention for a **retail event** (Black Friday by default) on a
single (product, seed) and contrast it with a matched **normal** day, producing:

  1. attention_neighbourhood_<variant>.pdf   — neighbourhood size across the test
        horizon, with the event day(s) marked (ties the attention story to the
        neighbour-count panel added to plot_results).
  2. attention_bar_<variant>_<tag>.pdf       — per-neighbour layer-1 attention
        (averaged over heads) with the raw similarity edge weight overlaid, so a
        reviewer sees the GAT did NOT simply copy the similarity prior.
  3. attention_egograph_<variant>_<tag>.pdf  — the target's ego-graph with each
        target→neighbour edge's width/colour ∝ its learned attention.
  4. attention_heads_<variant>_<tag>.pdf     — per-head attention heatmap
        (neighbours × heads) for layer 1, exposing head specialisation.
  5. attention_weights_<variant>_<tag>.csv   — neighbour id, category, similarity,
        per-head + mean layer-1 attention, layer-2 attention.

This module reuses the SAME data prep / graph build / model construction / training
the runner uses, imported from ``representation_quality_analysis`` (which itself
re-imports the runner's source modules), so the trained GAT here matches the real
experiments.  Attention is read with PyG's ``return_attention_weights=True`` on the
two ``EdgeWeightedGATConv`` layers — the returned ``alpha`` is the gated,
per-target-renormalised coefficient the model actually aggregates with.

Output: attention_interpretability/product_<id>_<store>/seed_<seed>/
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# All shared setup (paths/sys.path, constants, data prep, graph build, model
# construction, training fns, datasets) comes from the representation-quality
# module so the two analyses stay perfectly in sync with the runner.
import representation_quality_analysis as R


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
# Which GAT variant to interpret (must be a GAT head — attention is GAT-only).
GAT_VARIANT = 'gat_lstm'        # or 'gat_mlp'

# Retail event to snapshot.  Any binary exog flag works ('is_black_friday',
# 'is_christmas', 'is_thanksgiving', ...).  The chosen event day is the one whose
# ego-graph has the most neighbours (most to "explain"); a matched non-event day
# with the most neighbours is used as the contrast.
EVENT_FLAG = 'is_black_friday'

# Optional explicit override: a 'YYYY-MM-DD' string forces the event day instead
# of auto-picking from EVENT_FLAG.  None → auto.
EVENT_DATE_OVERRIDE = None

OUT_BASE = os.path.join(R.SCRIPT_DIR, "attention_interpretability")


# ════════════════════════════════════════════════════════════════════════════
#  Train the GAT model (mirrors representation_quality_analysis.run_variant)
# ════════════════════════════════════════════════════════════════════════════
def train_gat_model(variant, shared, prod, gb):
    """Build datasets from ``gb`` and train the GAT forecaster (z active)."""
    head      = R._variant_head(variant)
    exog_cols = R.EXOG_COLS_LSTM if head == 'lstm' else R.EXOG_COLS_MLP
    eb        = prod['exog_by_head'][head]

    if head == 'lstm':
        DatasetClass, collate_fn, train_fn = (
            R.GCN_LSTMTimeSeriesDataset, R.collate_pyg_ts_lstm, R.train_gcn_lstm_model)
    else:
        DatasetClass, collate_fn, train_fn = (
            R.GCNMLPTimeSeriesDataset, R.collate_pyg_ts_mlp, R.train_gcn_mlpmodel)

    pin = torch.cuda.is_available()
    tr_ds = DatasetClass(prod['train_scaled'], eb['train'] if exog_cols else None,
                         R.lookback_window, gb['pyg_train'], graph_window_size=R.WINDOW_SIZE)
    va_ds = DatasetClass(prod['val_scaled'], eb['val'] if exog_cols else None,
                         R.lookback_window, gb['pyg_val'], graph_window_size=R.WINDOW_SIZE)
    tr_ld = DataLoader(tr_ds, batch_size=R.BATCH_SIZE, shuffle=False, pin_memory=pin, collate_fn=collate_fn)
    va_ld = DataLoader(va_ds, batch_size=R.BATCH_SIZE, shuffle=False, pin_memory=pin, collate_fn=collate_fn)

    ts_input_size = 1 + (len(exog_cols) if exog_cols else 0)
    R._seed_everything(R.SELECTED_SEED)
    model = R.build_forecaster(variant, False, gb['in_channels'], ts_input_size).to(R.device)
    model.ablate_z = False
    lr  = R.resolve_learning_rate(variant)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    use_sched = R.USE_LR_SCHEDULER_LSTM if head == 'lstm' else R.USE_LR_SCHEDULER_MLP
    sch = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                      patience=R.PATIENCE // 3)
           if use_sched else None)
    model, *_ = train_fn(
        seed=R.SELECTED_SEED, epochs=R.EPOCHS, model=model,
        train_loader=tr_ld, val_loader=va_ld,
        criterion=nn.MSELoss(), criterion2=nn.MSELoss(),
        optimizer=opt, device=R.device, best_model_path=None,
        scheduler=sch, patience=R.PATIENCE, diag_csv_path=None, diag_meta={},
    )
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════════════════
#  Attention extraction
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_target_attention(model, data):
    """Run the two GAT layers on a single ego-graph and return the learned,
    edge-weight-gated attention into the target node (row 0).

    Returns ``(rows, n_heads)`` where ``rows`` is a list of dicts (one per
    neighbour + the self-loop) holding the layer-1 per-head + mean attention and
    the layer-2 attention.  Attention into the target sums to 1 over these rows
    (per layer / per head), since ``EdgeWeightedGATConv`` renormalises per target.
    """
    model.eval()
    x  = data.x.to(R.device)
    ei = data.edge_index.to(R.device)
    ea = (data.edge_attr.to(R.device)
          if (data.edge_attr is not None and data.edge_attr.numel() > 0) else None)

    # Layer 1 (multi-head, concat) — capture attention; then ELU + (eval) dropout.
    h1, (ei1, a1) = model.conv1(x, ei, edge_attr=ea, return_attention_weights=True)
    h1 = model.gnn_drop(model.activation(h1))
    # Layer 2 (single head, averaged) — capture attention on the same topology.
    _z, (ei2, a2) = model.conv2(h1, ei, edge_attr=ea, return_attention_weights=True)

    ei1 = ei1.cpu().numpy(); a1 = a1.detach().cpu().numpy()      # (2,E1), (E1,H)
    ei2 = ei2.cpu().numpy(); a2 = a2.detach().cpu().numpy()      # (2,E2), (E2,1)
    n_heads = a1.shape[1]

    node_order = list(getattr(data, 'node_order', range(data.num_nodes)))

    # layer-2 attention into target, keyed by source row (single head).
    l2_by_row = {}
    for e in range(ei2.shape[1]):
        if int(ei2[1, e]) == 0:
            l2_by_row[int(ei2[0, e])] = float(a2[e, 0])

    # Similarity edge weight target<->neighbour, from the ORIGINAL edges.
    sim_by_row = {}
    if data.edge_attr is not None and data.edge_attr.numel() > 0:
        oei = data.edge_index.cpu().numpy()
        oea = data.edge_attr.cpu().numpy().reshape(-1)
        for e in range(oei.shape[1]):
            s, t = int(oei[0, e]), int(oei[1, e])
            if t == 0 and s != 0:
                sim_by_row[s] = float(oea[e])

    rows = []
    for e in range(ei1.shape[1]):
        if int(ei1[1, e]) != 0:          # only edges aggregating INTO the target
            continue
        src = int(ei1[0, e])
        rows.append(dict(
            row=src,
            node=(node_order[src] if src < len(node_order) else f'row{src}'),
            is_self=(src == 0),
            att_l1_heads=a1[e].astype(float).tolist(),
            att_l1_mean=float(a1[e].mean()),
            att_l2=l2_by_row.get(src, float('nan')),
            similarity=(np.nan if src == 0 else sim_by_row.get(src, np.nan)),
        ))
    # Strongest attended first (self-loop sorts in naturally by its own weight).
    rows.sort(key=lambda r: r['att_l1_mean'], reverse=True)
    return rows, n_heads


# ════════════════════════════════════════════════════════════════════════════
#  Per-day ego-graph picking (reuses the runner-built, test-aligned graphs)
# ════════════════════════════════════════════════════════════════════════════
def neighbour_counts_over_test(gb):
    """#neighbours of the target ego-graph aligned to each test/forecast day.

    ``gb['pyg_future_graphs'][s]`` is the graph the model sees for test day ``s``
    (built from observed data here); ``num_nodes - 1`` is the direct-neighbour
    count.  Returns a numpy int array of length ``forecast_horizon``.
    """
    fut = gb['pyg_future_graphs']
    return np.array([int(g.num_nodes) - 1 for g in fut], dtype=int)


def _event_steps(prod, flag):
    """Test-window step indices (0..H-1) on which ``flag`` is set for the product."""
    df_p = prod['df_p']
    if flag not in df_p.columns:
        return []
    test_flags = df_p[flag].iloc[prod['test_slice']].to_numpy()
    return [int(s) for s in np.where(test_flags == 1)[0]]


def _step_date(prod, step):
    df_p = prod['df_p']
    return pd.Timestamp(df_p[R.DATE_COL].iloc[prod['test_start_idx'] + step])


def pick_days(prod, gb):
    """Choose the (event_step, normal_step) to interpret.

    event  : the EVENT_FLAG day (or EVENT_DATE_OVERRIDE) whose ego-graph has the
             most neighbours; falls back to the global max-neighbour day if the
             event is absent / isolated.
    normal : the non-event test day with the most neighbours (contrast).
    """
    counts = neighbour_counts_over_test(gb)
    df_p = prod['df_p']

    # candidate event steps
    if EVENT_DATE_OVERRIDE is not None:
        tgt = pd.Timestamp(EVENT_DATE_OVERRIDE).normalize()
        test_dates = pd.to_datetime(df_p[R.DATE_COL].iloc[prod['test_slice']].to_numpy())
        ev_steps = [int(s) for s, d in enumerate(test_dates) if pd.Timestamp(d).normalize() == tgt]
    else:
        ev_steps = _event_steps(prod, EVENT_FLAG)

    ev_steps = [s for s in ev_steps if counts[s] >= 1]
    if ev_steps:
        event_step = max(ev_steps, key=lambda s: counts[s])
        event_tag = EVENT_FLAG.replace('is_', '')
    else:
        event_step = int(np.argmax(counts))
        event_tag = 'maxneighbours'
        print(f"  [warn] no '{EVENT_FLAG}' day with neighbours in the test window; "
              f"using the global max-neighbour day instead.")

    # normal contrast: most-neighbour non-event day
    ev_all = set(_event_steps(prod, EVENT_FLAG))
    normal_candidates = [s for s in range(len(counts)) if s not in ev_all and counts[s] >= 1]
    normal_step = (max(normal_candidates, key=lambda s: counts[s])
                   if normal_candidates else None)

    return dict(counts=counts, event_step=event_step, event_tag=event_tag,
                normal_step=normal_step)


# ════════════════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════════════════
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_neighbourhood_evolution(prod, gb, picked, variant, out_dir):
    plt = _plt()
    counts = picked['counts']
    dates = pd.to_datetime(
        prod['df_p'][R.DATE_COL].iloc[prod['test_slice']].to_numpy())
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, counts, color='#1f77b4', lw=1.5, marker='o', ms=3,
            label='# neighbours')
    for s, c, lbl in [(picked['event_step'], 'red', f"event ({picked['event_tag']})"),
                      (picked['normal_step'], 'green', 'normal')]:
        if s is None:
            continue
        ax.axvline(dates[s], color=c, ls='--', lw=1.4, alpha=0.8,
                   label=f"{lbl}: {dates[s].date()} (n={counts[s]})")
    ax.set_title(f"Target neighbourhood size across the test horizon — "
                 f"{variant} | metric={R.METRIC}")
    ax.set_xlabel("Date"); ax.set_ylabel("# neighbours"); ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc='best')
    out = os.path.join(out_dir, f"attention_neighbourhood_{variant}.pdf")
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out}")


def plot_attention_bar(rows, n_heads, cat_by_id, title, out_path):
    plt = _plt()
    nbr = [r for r in rows if not r['is_self']]
    self_row = next((r for r in rows if r['is_self']), None)
    if not nbr:
        print(f"  [skip bar] no neighbours for {title}")
        return
    labels = [str(r['node']) for r in nbr]
    att    = [r['att_l1_mean'] for r in nbr]
    sim    = [r['similarity']  for r in nbr]

    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(nbr) + 3), 5))
    xs = np.arange(len(nbr))
    bars = ax.bar(xs, att, color='#4C72B0', label='Layer-1 attention (mean over heads)')
    if self_row is not None:
        ax.axhline(self_row['att_l1_mean'], color='gray', ls=':', lw=1.4,
                   label=f"self-attention = {self_row['att_l1_mean']:.2f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Attention into target')
    ax.set_xlabel('Neighbour item_id')

    # Similarity overlay on a twin axis — shows attention ≠ raw similarity.
    sim_arr = np.array(sim, dtype=float)
    if np.isfinite(sim_arr).any():
        ax2 = ax.twinx()
        ax2.plot(xs, sim_arr, color='#C44E52', marker='D', ms=5, lw=0,
                 label='Similarity edge weight')
        ax2.set_ylabel('Similarity edge weight', color='#C44E52')
        ax2.tick_params(axis='y', labelcolor='#C44E52')

    # annotate category under each bar if available
    if cat_by_id:
        for x, r in zip(xs, nbr):
            c = cat_by_id.get(r['node'])
            if c is not None:
                ax.annotate(str(c), (x, 0), textcoords="offset points",
                            xytext=(0, -28), ha='center', fontsize=6, color='dimgray',
                            rotation=45)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if np.isfinite(sim_arr).any() else ([], []))
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
    ax.set_title(title)
    fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out_path}")


def plot_attention_egograph(rows, target_id, title, out_path):
    plt = _plt()
    nbr = [r for r in rows if not r['is_self']]
    if not nbr:
        print(f"  [skip egograph] no neighbours for {title}")
        return
    att = np.array([r['att_l1_mean'] for r in nbr], dtype=float)
    amax = att.max() if att.max() > 0 else 1.0
    cmap = plt.get_cmap('viridis')

    n = len(nbr)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

    fig, ax = plt.subplots(figsize=(8, 8))
    for i, r in enumerate(nbr):
        x, y = pos[i]
        a = att[i]
        ax.plot([0, x], [0, y], color=cmap(a / amax),
                lw=1.0 + 7.0 * (a / amax), alpha=0.85, zorder=1, solid_capstyle='round')
        ax.scatter([x], [y], s=420, color=cmap(a / amax), edgecolors='black',
                   linewidths=0.8, zorder=2)
        ax.annotate(f"{r['node']}\n{a:.2f}", (x, y), ha='center', va='center',
                    fontsize=7, zorder=3)
    ax.scatter([0], [0], s=900, color='crimson', edgecolors='black',
               linewidths=1.5, zorder=4)
    ax.annotate(f"TARGET\n{target_id}", (0, 0), ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=amax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='Layer-1 attention (mean over heads)')
    ax.set_title(title); ax.set_aspect('equal'); ax.axis('off')
    lim = 1.35
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out_path}")


def plot_attention_heads(rows, n_heads, title, out_path):
    plt = _plt()
    nbr = [r for r in rows if not r['is_self']]
    if not nbr or n_heads <= 1:
        return
    mat = np.array([r['att_l1_heads'] for r in nbr], dtype=float)   # (n_nbr, H)
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * n_heads + 2),
                                    max(3, 0.4 * len(nbr) + 2)))
    im = ax.imshow(mat, aspect='auto', cmap='magma')
    ax.set_xticks(range(n_heads)); ax.set_xticklabels([f'h{j}' for j in range(n_heads)])
    ax.set_yticks(range(len(nbr)))
    ax.set_yticklabels([str(r['node']) for r in nbr], fontsize=7)
    ax.set_xlabel('Attention head'); ax.set_ylabel('Neighbour item_id')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Attention into target')
    fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out_path}")


def save_attention_csv(rows, n_heads, cat_by_id, out_path):
    recs = []
    for r in rows:
        rec = dict(node=r['node'], is_self=r['is_self'],
                   category=(cat_by_id.get(r['node']) if cat_by_id else None),
                   similarity=r['similarity'],
                   att_l1_mean=r['att_l1_mean'], att_l2=r['att_l2'])
        for j in range(n_heads):
            rec[f'att_l1_head{j}'] = r['att_l1_heads'][j]
        recs.append(rec)
    pd.DataFrame(recs).to_csv(out_path, index=False)
    print(f"  saved {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  One day's full attention snapshot
# ════════════════════════════════════════════════════════════════════════════
def snapshot_day(model, gb, prod, product_id, step, tag, cat_by_id, out_dir, variant):
    data = gb['pyg_future_graphs'][step]
    date = _step_date(prod, step)
    n_nbr = int(data.num_nodes) - 1
    print(f"\n--- {tag} day {date.date()} | step {step} | {n_nbr} neighbours ---")
    if n_nbr < 1:
        print("  no neighbours — nothing to interpret for this day.")
        return None

    rows, n_heads = extract_target_attention(model, data)
    title_core = (f"{variant} attention into target {product_id} — "
                  f"{tag} ({date.date()}) | metric={R.METRIC}")

    plot_attention_bar(
        rows, n_heads, cat_by_id, "Per-neighbour learned attention vs similarity\n" + title_core,
        os.path.join(out_dir, f"attention_bar_{variant}_{tag}.pdf"))
    plot_attention_egograph(
        rows, product_id, "Attention-weighted ego-graph\n" + title_core,
        os.path.join(out_dir, f"attention_egograph_{variant}_{tag}.pdf"))
    plot_attention_heads(
        rows, n_heads, "Per-head layer-1 attention\n" + title_core,
        os.path.join(out_dir, f"attention_heads_{variant}_{tag}.pdf"))
    save_attention_csv(
        rows, n_heads, cat_by_id,
        os.path.join(out_dir, f"attention_weights_{variant}_{tag}.csv"))

    # console summary: top-3 attended neighbours + similarity rank comparison
    nbr = [r for r in rows if not r['is_self']]
    sim_rank = {r['node']: i for i, r in enumerate(
        sorted(nbr, key=lambda r: (np.nan_to_num(r['similarity'], nan=-1)), reverse=True))}
    print("  top attended neighbours (att rank vs similarity rank):")
    for i, r in enumerate(nbr[:3]):
        print(f"    {i+1}. item {r['node']:<8} att={r['att_l1_mean']:.3f} "
              f"sim={r['similarity']:.3f} (similarity rank {sim_rank.get(r['node'], '?')})")
    return dict(date=date, n_neighbours=n_nbr, rows=rows)


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    if not GAT_VARIANT.startswith('gat'):
        raise ValueError(f"GAT_VARIANT must be a GAT head; got {GAT_VARIANT!r}")

    print(f"{'='*72}\nGAT ATTENTION INTERPRETABILITY — {GAT_VARIANT} | metric={R.METRIC} "
          f"| seed {R.SELECTED_SEED}\n{'='*72}")

    shared = R.prepare_shared()
    product_id, store_id = R._resolve_selected_product(shared)
    print(f"Selected product=({product_id}, {store_id})")
    prod = R.prepare_product(shared, product_id, store_id)
    if prod is None:
        raise RuntimeError(f"Product ({product_id},{store_id}) has insufficient history.")

    out_dir = os.path.join(OUT_BASE, f"product_{product_id}_{store_id}",
                           f"seed_{R.SELECTED_SEED}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nBuilding graphs + training {GAT_VARIANT} ...")
    gb = R.build_graphs_for_variant(GAT_VARIANT, shared, prod, product_id)
    model = train_gat_model(GAT_VARIANT, shared, prod, gb)

    cat_by_id = shared.get('cat_labels_dict', {}) or {}

    picked = pick_days(prod, gb)
    plot_neighbourhood_evolution(prod, gb, picked, GAT_VARIANT, out_dir)

    snapshot_day(model, gb, prod, product_id, picked['event_step'],
                 picked['event_tag'], cat_by_id, out_dir, GAT_VARIANT)
    if picked['normal_step'] is not None:
        snapshot_day(model, gb, prod, product_id, picked['normal_step'],
                     'normal', cat_by_id, out_dir, GAT_VARIANT)

    print(f"\nDone. Outputs under: {out_dir}")


if __name__ == '__main__':
    main()
