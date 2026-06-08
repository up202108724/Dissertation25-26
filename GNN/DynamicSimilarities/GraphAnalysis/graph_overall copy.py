import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make sibling imports work whether run from repo root or this folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.dirname(__file__))

from sklearn.preprocessing import StandardScaler

from utils import (
    compute_similarities_1vsAll,
    compute_distances_1vsAll,
)

sns.set_theme(style="whitegrid")


# =============================================================================
# Core: unified candidate collection + optional filtering
# =============================================================================
def collect_edge_weights(df_wide, item_ids, metric, metric_type,
                         window_size, step_size=1, train_end_idx=None,
                         mode="none", threshold=None, percentile=None,
                         enable_edges_within_star=True, verbose=True):
    """
    Returns
    -------
    all_weights : np.ndarray
    per_item    : dict[item_id] -> np.ndarray of kept (or all) values
    per_item_threshold : dict[item_id] -> float | None
        Effective threshold applied per item. For MODE='threshold' this is
        the supplied THRESHOLD; for MODE='percentile' it is the derived
        global top-k cutoff; for MODE='none' it is None.
    per_item_counts : dict[item_id] -> (n_candidates, n_star_kept, n_within_kept)
        For MODE='none', n_star_kept == n_candidates and n_within_kept == 0.
    """
    if mode not in ("none", "threshold", "percentile"):
        raise ValueError("mode must be 'none', 'threshold' or 'percentile'.")
    if mode == "threshold" and threshold is None:
        raise ValueError("threshold must be provided when mode='threshold'.")
    if mode == "percentile" and percentile is None:
        raise ValueError("percentile must be provided when mode='percentile'.")

    if metric_type == "similarity":
        compute_func = compute_similarities_1vsAll
    elif metric_type == "distance":
        compute_func = compute_distances_1vsAll
    else:
        raise ValueError("metric_type must be 'similarity' or 'distance'.")

    time_steps = df_wide.shape[1]
    scan_end = time_steps if train_end_idx is None else min(time_steps, train_end_idx)
    windows_range = list(range(0, scan_end - window_size + 1, step_size))

    per_item_kept = {}
    per_item_threshold = {}
    per_item_counts = {}
    all_kept = []

    def _pass_mask(values, cutoff):
        if metric_type == "distance":
            return values <= cutoff
        return values >= cutoff

    for item_id in item_ids:
        if item_id not in df_wide.index:
            print(f"[WARN] item_id {item_id} not in dataframe index. Skipping.")
            continue

        if verbose:
            tag = (f"threshold={threshold}" if mode == "threshold"
                   else f"percentile={percentile}%" if mode == "percentile"
                   else "no-filter")
            print(f"\n--- {item_id} | metric={metric} ({metric_type}) | "
                  f"window={window_size} | step={step_size} | "
                  f"mode={mode} | {tag} | within_star={enable_edges_within_star} | "
                  f"{len(windows_range)} windows ---")

        # ---- Pass 1: per-window compute (always required) ----
        # Cache the per-window data so we don't recompute in Pass 2 (percentile).
        per_window_cache = []
        all_candidates = []

        for start_idx in windows_range:
            end_idx = start_idx + window_size
            window_data = df_wide.iloc[:, start_idx:end_idx]

            target_ts = window_data.loc[item_id].values
            if np.sum(np.abs(target_ts)) == 0:
                per_window_cache.append(None)
                continue

            all_ts = window_data.values
            ids = window_data.index.values

            vals = compute_func(target_ts, all_ts, metric=metric)

            active_mask = np.sum(np.abs(all_ts), axis=1) > 0
            valid_mask = (ids != item_id) & active_mask
            valid_idx_all = np.where(valid_mask)[0]
            v_raw = vals[valid_mask]
            finite_mask = np.isfinite(v_raw)
            v = v_raw[finite_mask]
            valid_idx_all = valid_idx_all[finite_mask]

            per_window_cache.append((all_ts, v, valid_idx_all))
            all_candidates.append(v)

        all_candidates = (np.concatenate(all_candidates)
                          if all_candidates else np.array([], dtype=float))
        n_candidates = len(all_candidates)

        # ---- Determine the effective cutoff (if any) ----
        effective_cutoff = None
        if mode == "threshold":
            effective_cutoff = float(threshold)
        elif mode == "percentile" and n_candidates > 0:
            k = max(1, int(n_candidates * (percentile / 100.0)))
            if metric_type == "distance":
                effective_cutoff = float(np.sort(all_candidates)[:k][-1])
            else:
                effective_cutoff = float(np.sort(all_candidates)[::-1][:k][-1])
            if verbose:
                print(f"  derived global threshold @top {percentile}%: "
                      f"{effective_cutoff:.6f}")

        # ---- Build the kept pool ----
        kept = []
        n_star_kept = 0
        n_within_kept = 0

        if mode == "none":
            # No filtering: keep everything as "star"; no within-star pass.
            if n_candidates > 0:
                kept.append(all_candidates)
                n_star_kept = n_candidates
        else:
            for entry in per_window_cache:
                if entry is None:
                    continue
                all_ts, v, valid_idx_all = entry
                if len(v) == 0:
                    continue

                pm = _pass_mask(v, effective_cutoff)
                star_kept = v[pm]
                n_star_kept += len(star_kept)
                kept.append(star_kept)

                if enable_edges_within_star:
                    sel_idx = valid_idx_all[pm]
                    if len(sel_idx) > 1:
                        neigh_ts = all_ts[sel_idx]
                        for i in range(len(sel_idx) - 1):
                            sub = compute_func(neigh_ts[i],
                                               neigh_ts[i + 1:],
                                               metric=metric)
                            sub = sub[np.isfinite(sub)]
                            if len(sub) == 0:
                                continue
                            sub = sub[_pass_mask(sub, effective_cutoff)]
                            if len(sub) > 0:
                                n_within_kept += len(sub)
                                kept.append(sub)

        item_arr = (np.concatenate(kept) if kept else np.array([], dtype=float))
        per_item_kept[item_id] = item_arr
        per_item_threshold[item_id] = effective_cutoff
        per_item_counts[item_id] = (n_candidates, n_star_kept, n_within_kept)
        all_kept.extend(item_arr.tolist())

        if verbose:
            print(f"  candidates: {n_candidates:,} | star kept: {n_star_kept:,} | "
                  f"within-star kept: {n_within_kept:,} | total: {len(item_arr):,}")

    return (np.asarray(all_kept, dtype=float),
            per_item_kept, per_item_threshold, per_item_counts)


# =============================================================================
# Plotting / reporting
# =============================================================================
def report_distribution(all_weights, per_item_threshold, per_item_counts,
                        metric, metric_type, window_size, step_size,
                        mode, threshold, percentile, item_ids, out_dir,
                        percentiles=(1, 5, 10, 25, 50, 75, 90, 95, 99),
                        bins=100):
    os.makedirs(out_dir, exist_ok=True)

    if len(all_weights) == 0:
        print("[ERROR] No edge weights collected. Aborting.")
        return

    # --- Summary statistics ---
    summary = {
        'metric': metric,
        'metric_type': metric_type,
        'window_size': window_size,
        'step_size': step_size,
        'mode': mode,
        'threshold': threshold if mode == "threshold" else None,
        'percentile': percentile if mode == "percentile" else None,
        'n_items': len(item_ids),
        'n_edges': int(len(all_weights)),
        'mean': float(np.mean(all_weights)),
        'std': float(np.std(all_weights)),
        'median': float(np.median(all_weights)),
        'min': float(np.min(all_weights)),
        'max': float(np.max(all_weights)),
    }
    for p in percentiles:
        summary[f'P{p}'] = float(np.percentile(all_weights, p))

    header_tag = (f"threshold={threshold}" if mode == "threshold"
                  else f"top {percentile}%" if mode == "percentile"
                  else "no-filter")
    print(f"\n============== DISTRIBUTION ({header_tag}) ==============")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>12}: {v:.6f}")
        else:
            print(f"  {k:>12}: {v}")

    print("\n--- Per-item candidate/kept counts ---")
    for it, (nc, nsk, nwk) in per_item_counts.items():
        nk = nsk + nwk
        frac = (nk / nc * 100.0) if nc else 0.0
        th = per_item_threshold.get(it)
        th_str = f"th={th:.6f}" if isinstance(th, float) else "th=N/A"
        print(f"  {it}: {th_str} | candidates={nc:,} star_kept={nsk:,} "
              f"within_star_kept={nwk:,} total_kept={nk:,} ({frac:.3f}%)")

    # --- Persist artifacts ---
    if mode == "threshold":
        mode_suffix = f"th{threshold}"
    elif mode == "percentile":
        mode_suffix = f"pct{percentile}"
    else:
        mode_suffix = "all"
    tag = f"{metric}_{metric_type}_W{window_size}_S{step_size}_{mode_suffix}_{'_'.join(str(i) for i in item_ids)}"

    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, f"summary_{tag}.csv"), index=False)
    pd.DataFrame({
        'item_id': list(per_item_counts.keys()),
        'effective_threshold': [per_item_threshold.get(i) for i in per_item_counts],
        'n_candidates': [c[0] for c in per_item_counts.values()],
        'n_star_kept': [c[1] for c in per_item_counts.values()],
        'n_within_star_kept': [c[2] for c in per_item_counts.values()],
        'n_total_kept': [c[1] + c[2] for c in per_item_counts.values()],
    }).to_csv(os.path.join(out_dir, f"per_item_counts_{tag}.csv"), index=False)

    # --- Histogram plot ---
    plt.figure(figsize=(14, 6))
    sns.histplot(all_weights, bins=bins, kde=False, color='skyblue')
    plt.title(f"{header_tag.capitalize()} Distribution of Edge "
              f"{metric_type.capitalize()} ({metric}) | window={window_size}, "
              f"step={step_size} | {len(item_ids)} items, "
              f"{len(all_weights):,} edges")
    plt.xlabel(f"{metric_type.capitalize()} ({metric})")
    plt.ylabel("Frequency")

    if mode == "threshold":
        plt.axvline(threshold, color='black', linestyle='-', linewidth=2,
                    label=f"Threshold ({threshold})")
    plt.axvline(summary['mean'], color='red', linestyle='--',
                label=f"Mean ({summary['mean']:.3f})")
    plt.axvline(summary['median'], color='green', linestyle='-',
                label=f"Median ({summary['median']:.3f})")

    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(out_dir, f"hist_{tag}.png")
    plt.savefig(hist_path, dpi=140)
    plt.close()
    print(f"\nSaved histogram -> {hist_path}")

    # --- Boxplot ---
    plt.figure(figsize=(10, 3))
    sns.boxplot(x=all_weights, color='lightblue', fliersize=1)
    plt.title(f"Boxplot ({header_tag}) of Edge "
              f"{metric_type.capitalize()} ({metric}) | window={window_size}")
    plt.xlabel(f"{metric_type.capitalize()}")
    plt.tight_layout()
    box_path = os.path.join(out_dir, f"box_{tag}.png")
    plt.savefig(box_path, dpi=140)
    plt.close()
    print(f"Saved boxplot   -> {box_path}")

    return summary


def plot_per_item_comparison(per_item, metric, metric_type, window_size,
                             step_size, mode, threshold, percentile,
                             out_dir, bins=80):
    if not per_item:
        return
    os.makedirs(out_dir, exist_ok=True)

    if mode == "threshold":
        mode_suffix = f"th{threshold}"
        header = f"threshold={threshold}"
    elif mode == "percentile":
        mode_suffix = f"pct{percentile}"
        header = f"top {percentile}%"
    else:
        mode_suffix = "all"
        header = "no-filter"
    tag = f"{metric}_{metric_type}_W{window_size}_S{step_size}_{mode_suffix}_{'_'.join(str(i) for i in per_item.keys())}"
    for item_id, vals in per_item.items():
        if len(vals) == 0:
            continue
        sns.kdeplot(vals, label=str(item_id), fill=False, alpha=0.8)
    plt.title(f"Per-Item KDE ({header}) of Edge "
              f"{metric_type.capitalize()} ({metric}) | window={window_size}")
    plt.xlabel(f"{metric_type.capitalize()}")
    plt.legend(title="item_id", fontsize=8)
    plt.tight_layout()
    p = os.path.join(out_dir, f"per_item_kde_{tag}.png")
    plt.savefig(p, dpi=140)
    plt.close()
    print(f"Saved per-item KDE -> {p}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # ---------------- Configuration ----------------
    # MODE: 'none' | 'threshold' | 'percentile'
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, SCRIPT_DIR)                                          # local modules (inference, train, ...)
    sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '../..')))  # DynamicSimilarities/ (plots, utils)

    MODE = "percentile"             # 'none' | 'threshold' | 'percentile'
    METRIC = "spearman"               # 'pearson' | 'spearman' | 'kendall' |
                                 # 'cid' | 'dtw' | 'manhattan' | ...
    METRIC_TYPE = "similarity"     # 'similarity' or 'distance'
    WINDOW_SIZE = 30
    STEP_SIZE = 1

    THRESHOLD = 0.70              # used iff MODE == 'threshold'
    PERCENTILE = 1               # used iff MODE == 'percentile'
    ENABLE_EDGES_WITHIN_STAR = True   # ignored when MODE == 'none'
    TOTAL_TIME_STEPS = 761
    FORECAST_HORIZON = 153
    VAL_SIZE = 31
    TRAIN_SIZE = TOTAL_TIME_STEPS - VAL_SIZE - FORECAST_HORIZON
    USE_TRAIN_VAL_ONLY = True
    # For distance metrics on raw counts (CID/DTW/Euclidean/...) values are
    # dominated by magnitude. Z-normalize each item using train slice stats
    # (matches graph_plotting_statistics.ipynb).
    ENABLE_Z_NORMALIZATION = True

    OUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "OverallDistributions")

    # ---------------- Load dataset ----------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FULL_DATA_PATH = os.path.join(BASE_DIR, '..', '..', '..',
                                  'dataset', 'data_andre_classified.feather')
    DATA_PATH = os.path.join(BASE_DIR, '..', '..', '..',
                             'dataset', 'top_12500.feather')

    # Focal products: the 61 items in DATA_PATH
    print(f"Loading focal products from {DATA_PATH}...")
    df_focal = pd.read_feather(DATA_PATH)
    ITEM_IDS = df_focal['item_id'].unique().tolist()
    print(f"Focal products: {len(ITEM_IDS)}")

    # Neighbour pool: full dataset (all products serve as candidate neighbours)
    print(f"Loading full neighbour dataset from {FULL_DATA_PATH}...")
    df_full = pd.read_feather(FULL_DATA_PATH)

    df_wide = (df_full.pivot_table(index='item_id', columns='date',
                                   values='value', aggfunc='sum')
                      .fillna(0))
    print(f"Full dataset: {df_wide.shape[0]} products, {df_wide.shape[1]} time steps")

    if USE_TRAIN_VAL_ONLY:
        df_wide = df_wide.iloc[:, :TRAIN_SIZE + VAL_SIZE]

    train_end_idx = (TRAIN_SIZE + VAL_SIZE) if USE_TRAIN_VAL_ONLY else None
    print(f"Dataframe shape (rows x time_steps): {df_wide.shape}")

    # ---------------- Optional Z-normalization (per item, fit on train) ----------------
    if ENABLE_Z_NORMALIZATION and METRIC_TYPE == "distance":
        print("Applying per-item Z-score normalization (fit on train slice)...")
        train_slice = df_wide.iloc[:, :TRAIN_SIZE]
        df_wide_scaled = df_wide.copy()
        for it in df_wide.index:
            scaler = StandardScaler()
            scaler.fit(train_slice.loc[it].values.reshape(-1, 1))
            df_wide_scaled.loc[it] = scaler.transform(
                df_wide.loc[it].values.reshape(-1, 1)
            ).flatten()
        df_wide = df_wide_scaled

    # ---------------- Output dir per mode ----------------
    if MODE == "threshold":
        mode_dir = f"W{WINDOW_SIZE}_S{STEP_SIZE}_th{THRESHOLD}"
    elif MODE == "percentile":
        mode_dir = f"W{WINDOW_SIZE}_S{STEP_SIZE}_pct{PERCENTILE}"
    else:
        mode_dir = f"W{WINDOW_SIZE}_S{STEP_SIZE}_all"

    out_base_dir = os.path.join(OUT_BASE, f"{METRIC}_{METRIC_TYPE}", MODE, mode_dir)

    # ---------------- Per-product loop ----------------
    for item_id in ITEM_IDS:
        print(f"\n{'='*70}")
        print(f"Processing item_id={item_id} ({ITEM_IDS.index(item_id)+1}/{len(ITEM_IDS)})")
        print(f"{'='*70}")

        all_weights, per_item, per_item_th, per_item_counts = collect_edge_weights(
            df_wide=df_wide,
            item_ids=[item_id],
            metric=METRIC,
            metric_type=METRIC_TYPE,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
            train_end_idx=train_end_idx,
            mode=MODE,
            threshold=THRESHOLD,
            percentile=PERCENTILE,
            enable_edges_within_star=ENABLE_EDGES_WITHIN_STAR,
            verbose=True,
        )

        if len(all_weights) == 0:
            print(f"  [SKIP] No edges collected for item_id={item_id}.")
            continue

        out_dir = os.path.join(out_base_dir, str(item_id))
        report_distribution(
            all_weights=all_weights,
            per_item_threshold=per_item_th,
            per_item_counts=per_item_counts,
            metric=METRIC,
            metric_type=METRIC_TYPE,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
            mode=MODE,
            threshold=THRESHOLD,
            percentile=PERCENTILE,
            item_ids=[item_id],
            out_dir=out_dir,
        )

    print(f"\nAll artifacts saved under: {out_base_dir}")
