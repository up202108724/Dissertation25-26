import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import gaussian_kde

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.dirname(__file__))

from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")


# =============================================================================
# Vectorised focal-vs-full matrix  (one GPU pass per window)
# =============================================================================
def _rank_rows(X, device):
    """Replace each row with its rank vector (no tie correction needed here)."""
    _, idx = torch.sort(X, dim=1)
    R = torch.empty_like(X)
    R.scatter_(1, idx,
               torch.arange(1, X.shape[1] + 1, dtype=torch.float32, device=device)
               .unsqueeze(0).expand_as(X))
    return R


def _focal_vs_full_matrix(focal_ts, full_ts, self_col_indices, metric, eps=1e-12):
    """
    Compute the (N_focal × N_full) pairwise matrix and return a flat 1-D array,
    with self-pairs (focal product vs itself in the full set) removed.

    Parameters
    ----------
    focal_ts        : (N_focal, W) numpy array  – active focal product windows
    full_ts         : (N_full,  W) numpy array  – active full-dataset windows
    self_col_indices: list of length N_focal – column index in full_ts that
                      corresponds to each focal product (-1 if not present)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    F = torch.tensor(focal_ts, dtype=torch.float32, device=device)  # (N_focal, W)
    X = torch.tensor(full_ts,  dtype=torch.float32, device=device)  # (N_full,  W)

    if metric in ('spearman', 'pearson'):
        if metric == 'spearman':
            F = _rank_rows(F, device)
            X = _rank_rows(X, device)
        F = F - F.mean(dim=1, keepdim=True)
        X = X - X.mean(dim=1, keepdim=True)
        cov   = F @ X.T                                        # (N_focal, N_full)
        std_F = torch.sqrt((F ** 2).sum(dim=1, keepdim=True))  # (N_focal, 1)
        std_X = torch.sqrt((X ** 2).sum(dim=1, keepdim=True))  # (N_full,  1)
        mat   = cov / (std_F * std_X.T + eps)

    elif metric == 'kendall':
        W = F.shape[1]
        i1, i2    = torch.triu_indices(W, W, offset=1, device=device)
        signs_F   = torch.sign(F[:, i1] - F[:, i2])           # (N_focal, P)
        signs_X   = torch.sign(X[:, i1] - X[:, i2])           # (N_full,  P)
        S         = signs_F @ signs_X.T                        # (N_focal, N_full)
        nt_F      = (signs_F ** 2).sum(dim=1, keepdim=True)    # (N_focal, 1)
        nt_X      = (signs_X ** 2).sum(dim=1, keepdim=True)    # (N_full,  1)
        denom     = torch.sqrt(nt_F * nt_X.T)
        mat       = torch.where(denom == 0,
                                torch.zeros_like(S, dtype=torch.float32),
                                S.float() / denom)

    elif metric == 'cid':
        ed   = torch.cdist(F, X, p=2)                          # (N_focal, N_full)
        ce_F = torch.sqrt((torch.diff(F, dim=1) ** 2).sum(dim=1))  # (N_focal,)
        ce_X = torch.sqrt((torch.diff(X, dim=1) ** 2).sum(dim=1))  # (N_full,)
        ce_F = torch.clamp(ce_F, min=eps)
        ce_X = torch.clamp(ce_X, min=eps)
        cf   = (torch.maximum(ce_F.unsqueeze(1), ce_X.unsqueeze(0)) /
                torch.minimum(ce_F.unsqueeze(1), ce_X.unsqueeze(0)))
        mat  = ed * cf

    else:
        raise ValueError(f"Metric '{metric}' not supported. "
                         f"Supported: spearman, kendall, pearson, cid.")

    mat_np = mat.cpu().numpy()

    # Mask self-pairs (focal product vs itself in the full column set)
    for f_row, col in enumerate(self_col_indices):
        if 0 <= col < mat_np.shape[1]:
            mat_np[f_row, col] = np.nan

    flat = mat_np.ravel()
    return flat[np.isfinite(flat)]


# =============================================================================
# Collection loop
# =============================================================================
def collect_focal_vs_full_weights(df_wide, focal_ids, metric,
                                   window_size, step_size=1,
                                   train_end_idx=None, verbose=True):
    """
    For every sliding window computes the (N_focal × N_full) pairwise matrix
    — focal products (rows) vs the complete product catalog (columns) — and
    returns a flat array of all edge-weight candidates.

    This mirrors exactly what happens during graph construction: each focal
    product is compared against every other product in the dataset.
    """
    all_ids   = df_wide.index.tolist()
    id_to_col = {pid: c for c, pid in enumerate(all_ids)}

    valid_focal = [i for i in focal_ids if i in id_to_col]
    if not valid_focal:
        raise ValueError("None of the focal_ids found in df_wide.")

    time_steps = df_wide.shape[1]
    scan_end   = time_steps if train_end_idx is None else min(time_steps, train_end_idx)
    windows    = list(range(0, scan_end - window_size + 1, step_size))

    n_focal = len(valid_focal)
    n_full  = len(all_ids)

    if verbose:
        print(f"\n--- Focal-vs-Full | metric={metric} | "
              f"window={window_size} | step={step_size} | "
              f"{n_focal} focal × {n_full} full | "
              f"{len(windows)} windows ---")
        print(f"  Expected observations: ~{n_focal * (n_full - 1) * len(windows):,}")

    # Pre-build index arrays for fast slicing
    focal_row_idx = np.array([id_to_col[i] for i in valid_focal])

    full_mat_all = df_wide.values  # (N_full, T) — slice columns per window

    all_values = []

    for w_idx, start_idx in enumerate(windows):
        window_mat = full_mat_all[:, start_idx:start_idx + window_size]  # (N_full, W)

        # Active mask over full set
        full_active_mask = np.sum(np.abs(window_mat), axis=1) > 0
        active_full_ts   = window_mat[full_active_mask]                  # (M_full, W)

        # Which focal products are active?
        focal_active_in_full_mask = full_active_mask[focal_row_idx]
        active_focal_rows         = focal_row_idx[focal_active_in_full_mask]
        active_focal_ts           = window_mat[active_focal_rows]        # (M_focal, W)

        if active_focal_ts.shape[0] == 0 or active_full_ts.shape[0] < 2:
            continue

        # Recompute self-column indices within the *active* full sub-array
        # active_full_ts rows correspond to original rows where full_active_mask is True
        active_full_orig_rows = np.where(full_active_mask)[0]
        orig_to_active_col    = {orig: new for new, orig in enumerate(active_full_orig_rows)}
        self_cols = [orig_to_active_col.get(r, -1) for r in active_focal_rows]

        vals = _focal_vs_full_matrix(active_focal_ts, active_full_ts,
                                     self_cols, metric)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            all_values.append(vals)

        if verbose and (w_idx + 1) % 50 == 0:
            total_so_far = sum(len(v) for v in all_values)
            print(f"  Window {w_idx + 1}/{len(windows)} | "
                  f"collected {total_so_far:,} values so far")

    if not all_values:
        return np.array([], dtype=float)

    return np.concatenate(all_values)


# =============================================================================
# Plotting  (pure matplotlib — avoids seaborn's pandas OOM on large arrays)
# =============================================================================
def plot_distribution(all_values, metric, metric_type, window_size,
                      step_size, n_focal, n_full, out_dir,
                      percentile_markers=(25, 50, 75, 90, 95),
                      bins=150, kde_sample=200_000,
                      cutoff=None, top_percentile=None):
    os.makedirs(out_dir, exist_ok=True)

    if len(all_values) == 0:
        print("[ERROR] No values collected. Aborting.")
        return

    stats = {
        'metric':         metric,
        'metric_type':    metric_type,
        'window_size':    window_size,
        'step_size':      step_size,
        'n_focal':        n_focal,
        'n_full':         n_full,
        'n_observations': int(len(all_values)),
        'mean':           float(np.mean(all_values)),
        'std':            float(np.std(all_values)),
        'median':         float(np.median(all_values)),
        'min':            float(np.min(all_values)),
        'max':            float(np.max(all_values)),
    }
    pct_vals = {}
    for p in percentile_markers:
        pct_vals[p] = float(np.percentile(all_values, p))
        stats[f'P{p}'] = pct_vals[p]

    print(f"\n============== Distribution ({metric}) ==============")
    for k, v in stats.items():
        print(f"  {k:>16}: {v:.6f}" if isinstance(v, float) else f"  {k:>16}: {v}")

    pct_suffix = f"_top{top_percentile}pct" if top_percentile is not None else ""
    tag = f"focal_vs_full_{metric}_{metric_type}_W{window_size}_S{step_size}{pct_suffix}"
    pd.DataFrame([stats]).to_csv(os.path.join(out_dir, f"summary_{tag}.csv"), index=False)

    pct_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(percentile_markers)))

    # --- Histogram (numpy binning → no pandas overhead) ---
    fig, ax = plt.subplots(figsize=(14, 6))

    counts, edges = np.histogram(all_values, bins=bins)
    width = edges[1] - edges[0]
    ax.bar(edges[:-1], counts, width=width, align='edge',
           color='steelblue', alpha=0.55, edgecolor='none', label='_nolegend_')

    # KDE on a random sample to keep it fast
    sample = (all_values if len(all_values) <= kde_sample
              else np.random.choice(all_values, kde_sample, replace=False))
    kde_fn = gaussian_kde(sample, bw_method='scott')
    x_kde  = np.linspace(all_values.min(), all_values.max(), 600)
    ax2    = ax.twinx()
    ax2.plot(x_kde, kde_fn(x_kde), color='steelblue', linewidth=1.6, label='KDE')
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_ylim(bottom=0)

    ax.axvline(stats['mean'],   color='red',   linestyle='--', linewidth=1.8,
               label=f"Mean ({stats['mean']:.3f})")
    ax.axvline(stats['median'], color='green', linestyle='-',  linewidth=1.8,
               label=f"Median ({stats['median']:.3f})")
    for p, color in zip(percentile_markers, pct_colors):
        ax.axvline(pct_vals[p], color=color, linestyle=':', linewidth=1.3,
                   label=f"P{p} ({pct_vals[p]:.3f})")
    if cutoff is not None:
        ax.axvline(cutoff, color='black', linestyle='-', linewidth=2.0,
                   label=f"Threshold ({cutoff:.3f})")

    title_suffix = f" | top {top_percentile}%" if top_percentile is not None else ""
    ax.set_title(
        f"Focal-vs-Full Pairwise {metric_type.capitalize()} Distribution — {metric}{title_suffix} | "
        f"window={window_size} d, step={step_size} | "
        f"{n_focal} focal × {n_full} full | {len(all_values):,} observations",
        fontsize=11,
    )
    ax.set_xlabel(f"{metric_type.capitalize()} ({metric})", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    hist_path = os.path.join(out_dir, f"hist_{tag}.png")
    hist_pdf  = os.path.join(out_dir, f"hist_{tag}.pdf")
    plt.savefig(hist_path, dpi=150)
    plt.savefig(hist_pdf)
    plt.close()
    print(f"Saved histogram -> {hist_path}")
    print(f"Saved histogram -> {hist_pdf}")

    # --- Boxplot (matplotlib native) ---
    _, ax = plt.subplots(figsize=(10, 3))
    ax.boxplot(all_values, vert=False,
               flierprops=dict(marker='.', markersize=1, alpha=0.3),
               medianprops=dict(color='green', linewidth=2),
               boxprops=dict(color='steelblue'))
    ax.set_title(f"Boxplot Focal-vs-Full ({metric}){title_suffix} | window={window_size} days")
    ax.set_xlabel(f"{metric_type.capitalize()} ({metric})", fontsize=11)
    plt.tight_layout()
    box_path = os.path.join(out_dir, f"box_{tag}.png")
    box_pdf  = os.path.join(out_dir, f"box_{tag}.pdf")
    plt.savefig(box_path, dpi=150)
    plt.savefig(box_pdf)
    plt.close()
    print(f"Saved boxplot   -> {box_path}")
    print(f"Saved boxplot   -> {box_pdf}")

    return stats


# =============================================================================
# Node-count analysis by season
# =============================================================================
def _holiday_tag(date):
    """Return 'thanksgiving', 'christmas', 'new_year_eve', or None."""
    m, d = date.month, date.day
    if (m == 11 and d >= 20) or (m == 12 and d <= 1):
        return 'thanksgiving'
    if m == 12 and 18 <= d <= 27:
        return 'christmas'
    if (m == 12 and d >= 28) or (m == 1 and d <= 2):
        return 'new_year_eve'
    return None


def compute_node_stats(df_wide, window_size, step_size=1,
                       train_end_idx=None, out_dir=None):
    """
    For each sliding window count the number of active nodes (products with
    at least one non-zero value in the window).  Tag windows whose *end date*
    falls inside Thanksgiving (Nov 20 – Dec 1), Christmas (Dec 18–27), or
    New Year Eve (Dec 28 – Jan 2) seasons.

    Writes two CSVs when ``out_dir`` is supplied:
      node_stats_per_window.csv  – one row per window
      node_stats_summary.csv     – overall and per-season averages
    """
    cols = pd.to_datetime(df_wide.columns)
    time_steps = df_wide.shape[1]
    scan_end = time_steps if train_end_idx is None else min(time_steps, train_end_idx)
    windows = list(range(0, scan_end - window_size + 1, step_size))

    values_mat = df_wide.values  # (N_items, T)

    rows = []
    for start_idx in windows:
        end_idx    = start_idx + window_size - 1
        window_mat = values_mat[:, start_idx : start_idx + window_size]
        n_active   = int((np.abs(window_mat).sum(axis=1) > 0).sum())

        start_date = cols[start_idx]
        end_date   = cols[end_idx]

        # Tag by end date; fall back to start date for windows straddling a boundary
        tag = _holiday_tag(end_date) or _holiday_tag(start_date)

        rows.append({
            'window_start':   start_date.date(),
            'window_end':     end_date.date(),
            'n_active_nodes': n_active,
            'holiday_tag':    tag or '',
        })

    df_win  = pd.DataFrame(rows)
    is_hol  = df_win['holiday_tag'] != ''
    non_hol = df_win.loc[~is_hol, 'n_active_nodes']

    summary = {
        'avg_nodes_total':           round(float(df_win['n_active_nodes'].mean()), 4),
        'avg_nodes_outside_holiday': round(float(non_hol.mean()), 4) if len(non_hol) else float('nan'),
        'n_windows_total':           len(df_win),
        'n_windows_holiday':         int(is_hol.sum()),
        'n_windows_non_holiday':     int((~is_hol).sum()),
    }
    for season in ('thanksgiving', 'christmas', 'new_year_eve'):
        mask = df_win['holiday_tag'] == season
        vals = df_win.loc[mask, 'n_active_nodes']
        summary[f'avg_nodes_{season}'] = round(float(vals.mean()), 4) if mask.any() else float('nan')
        summary[f'n_windows_{season}'] = int(mask.sum())

    print("\n====== Node Count Summary by Season ======")
    for k, v in summary.items():
        print(f"  {k:<40}: {v}")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        df_win.to_csv(os.path.join(out_dir, 'node_stats_per_window.csv'), index=False)
        pd.DataFrame([summary]).to_csv(
            os.path.join(out_dir, 'node_stats_summary.csv'), index=False)
        print(f"Saved node_stats_per_window.csv + node_stats_summary.csv -> {out_dir}")

    return summary, df_win


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, SCRIPT_DIR)
    sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '../..')))

    # ---- Configuration ----------------------------------------
    METRIC          = "cid"   # 'spearman' | 'kendall' | 'cid'
    METRIC_TYPE     = "distance" # 'similarity' | 'distance'  (cid → 'distance')
    WINDOW_SIZE     = 30
    STEP_SIZE       = 1
    TOP_PERCENTILE  = 1            # plot a second chart zoomed into the top N% strongest pairs

    TOTAL_TIME_STEPS       = 761
    FORECAST_HORIZON       = 153
    VAL_SIZE               = 61
    TRAIN_SIZE             = TOTAL_TIME_STEPS - VAL_SIZE - FORECAST_HORIZON
    USE_TRAIN_VAL_ONLY     = True
    ENABLE_Z_NORMALIZATION = True   # applied only when METRIC_TYPE == 'distance'

    OUT_BASE = os.path.join(SCRIPT_DIR, "AllvAllDistributions")

    # ---- Load data --------------------------------------------
    BASE_DIR       = SCRIPT_DIR
    FULL_DATA_PATH = os.path.join(BASE_DIR, '..', '..', '..',
                                  'dataset', 'data_andre_classified.feather')
    DATA_PATH      = os.path.join(BASE_DIR, '..', '..', '..',
                                  'dataset', 'top_12500.feather')

    print(f"Loading full dataset from {FULL_DATA_PATH}...")
    df_full = pd.read_feather(FULL_DATA_PATH)
    df_wide = (df_full.pivot_table(index='item_id', columns='date',
                                   values='value', aggfunc='sum')
                      .fillna(0))
    print(f"Full dataset: {df_wide.shape[0]} products × {df_wide.shape[1]} time steps")

    N_FOCAL = 60
    print(f"Loading focal products from {DATA_PATH}...")
    df_focal  = pd.read_feather(DATA_PATH)
    _focal_totals = (df_focal.groupby('item_id')['value'].sum()
                              .sort_values(ascending=False))
    focal_ids = _focal_totals.head(N_FOCAL).index.tolist()
    print(f"Focal products: {len(focal_ids)} (top {N_FOCAL} by total sales, "
          f"from {_focal_totals.shape[0]} candidates)")

    if USE_TRAIN_VAL_ONLY:
        df_wide = df_wide.iloc[:, :TRAIN_SIZE + VAL_SIZE]

    train_end_idx = (TRAIN_SIZE + VAL_SIZE) if USE_TRAIN_VAL_ONLY else None
    print(f"Dataframe shape used: {df_wide.shape}")

    if ENABLE_Z_NORMALIZATION and METRIC_TYPE == "distance":
        print("Applying per-item Z-score normalization (fit on train slice)...")
        train_slice    = df_wide.iloc[:, :TRAIN_SIZE]
        df_wide_scaled = df_wide.copy()
        for it in df_wide.index:
            scaler = StandardScaler()
            scaler.fit(train_slice.loc[it].values.reshape(-1, 1))
            df_wide_scaled.loc[it] = scaler.transform(
                df_wide.loc[it].values.reshape(-1, 1)
            ).flatten()
        df_wide = df_wide_scaled

    # ---- Compute focal-vs-full --------------------------------
    all_values = collect_focal_vs_full_weights(
        df_wide       = df_wide,
        focal_ids     = focal_ids,
        metric        = METRIC,
        window_size   = WINDOW_SIZE,
        step_size     = STEP_SIZE,
        train_end_idx = train_end_idx,
        verbose       = True,
    )

    print(f"\nTotal observations collected: {len(all_values):,}")

    out_dir = os.path.join(
        OUT_BASE,
        f"{METRIC}_{METRIC_TYPE}_W{WINDOW_SIZE}_S{STEP_SIZE}_focal_vs_full",
    )

    # ---- Node-count stats by season --------------------------
    compute_node_stats(
        df_wide       = df_wide,
        window_size   = WINDOW_SIZE,
        step_size     = STEP_SIZE,
        train_end_idx = train_end_idx,
        out_dir       = out_dir,
    )

    # --- Full distribution ---
    plot_distribution(
        all_values  = all_values,
        metric      = METRIC,
        metric_type = METRIC_TYPE,
        window_size = WINDOW_SIZE,
        step_size   = STEP_SIZE,
        n_focal     = len(focal_ids),
        n_full      = df_wide.shape[0],
        out_dir     = out_dir,
    )

    # --- Top-N% distribution (strongest pairs only) ---
    if METRIC_TYPE == "similarity":
        cutoff     = float(np.percentile(all_values, 100 - TOP_PERCENTILE))
        top_values = all_values[all_values >= cutoff]
    else:
        cutoff     = float(np.percentile(all_values, TOP_PERCENTILE))
        top_values = all_values[all_values <= cutoff]

    print(f"\nTop {TOP_PERCENTILE}% cutoff: {cutoff:.6f} | "
          f"{len(top_values):,} observations retained")

    plot_distribution(
        all_values     = top_values,
        metric         = METRIC,
        metric_type    = METRIC_TYPE,
        window_size    = WINDOW_SIZE,
        step_size      = STEP_SIZE,
        n_focal        = len(focal_ids),
        n_full         = df_wide.shape[0],
        out_dir        = out_dir,
        cutoff         = cutoff,
        top_percentile = TOP_PERCENTILE,
    )

    print(f"\nAll artifacts saved under: {out_dir}")
