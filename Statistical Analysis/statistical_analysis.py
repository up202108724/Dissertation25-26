import os
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from itertools import combinations


# Metrics where a larger value is better. Everything else (rmse, mae, ...) is
# treated as lower-is-better when ranking models.
HIGHER_IS_BETTER = {'pocid', 'r2_score', 'r2'}


def _higher_is_better(metric):
    return metric.lower() in HIGHER_IS_BETTER


def run_demsar_framework(data_input, metric='rmse', alpha=0.05, mode='total', seed_value=None, compare_ablations=False):
    """
    Executes the Demšar statistical framework (Friedman + Nemenyi).
    compare_ablations: If True, treats ablate_z=True rows as separate models. 
                       If False, filters them out to only evaluate full models.
    """
    if mode == 'per_seed' and seed_value is None:
        raise ValueError("You must provide a 'seed_value' when using mode='per_seed'.")

    # 1. Load Data
    if isinstance(data_input, str):
        df = pd.read_csv(data_input)
    elif isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise ValueError("data_input must be a file path string or a pandas DataFrame")

    # ---------------------------------------------------------
    # NEW: Handle Ablations elegantly
    # ---------------------------------------------------------
    if compare_ablations:
        # Identify rows where ablate_z is True
        is_ablated = df['ablate_z'] == True
        # Rename them so the pivot table treats them as a brand new model
        df.loc[is_ablated, 'model_variant'] = df.loc[is_ablated, 'model_variant'] + '_ablated'
        print(f"[*] ABLATION MODE: Treating ablate_z=True as separate models.")
    else:
        # Filter OUT ablated runs. Keep only ablate_z=False OR baselines (which are NaN)
        df = df[(df['ablate_z'] == False) | (df['ablate_z'].isna())]
        print(f"[*] STANDARD MODE: Filtered out ablate_z=True runs.")

    # 2. Handle Modes (Filtering and Block Selection)
    if mode == 'per_seed':
        print(f"\n{'='*60}")
        print(f"--- Running Statistical Evaluation for {metric.upper()} | SEED: {seed_value} ---")
        print(f"{'='*60}")
        df = df[df['seed'] == seed_value]
        block_cols = ['product_id']
        pivot_agg = 'first'
    elif mode == 'per_product':
        # Conservative blocking: one block per product (metric averaged over seeds).
        print(f"\n{'='*60}")
        print(f"--- Running Statistical Evaluation for {metric.upper()} | PER-PRODUCT (mean over seeds) ---")
        print(f"{'='*60}")
        block_cols = ['product_id']
        pivot_agg = 'mean'
    else:
        print(f"\n{'='*60}")
        print(f"--- Running Statistical Evaluation for {metric.upper()} | TOTAL SEEDS ---")
        print(f"{'='*60}")
        block_cols = ['product_id', 'seed']
        pivot_agg = 'first'

    # 3. Pivot the data to get blocks
    pivot_df = df.pivot_table(
        index=block_cols,
        columns='model_variant',
        values=metric,
        aggfunc=pivot_agg
    )
    
    initial_rows = len(pivot_df)
    pivot_df = pivot_df.dropna()
    print(f"Loaded {initial_rows} blocks.")
    print(f"Retained {len(pivot_df)} complete blocks after dropping missing runs.\n")
    
    if len(pivot_df) < 2:
        print("Not enough complete blocks to run statistical tests.")
        return None, None, None

    models = pivot_df.columns.tolist()
    
    # 4. Calculate Average Ranks
    # Rank 1 = best. POCID / R2 are higher-is-better, error metrics lower-is-better.
    ranks = pivot_df.rank(axis=1, method='average', ascending=not _higher_is_better(metric))
    avg_ranks = ranks.mean().sort_values()
    
    print("--- Average Ranks (Lower is Better) ---")
    for model, rank in avg_ranks.items():
        print(f"{model:<25} : {rank:.3f}")
    print("\n")
    
    # 5. Omnibus Friedman Test
    data_arrays = [pivot_df[model].values for model in models]
    stat, p_value = stats.friedmanchisquare(*data_arrays)
    
    print("--- Friedman Test Results ---")
    print(f"Test Statistic: {stat:.4f}")
    print(f"P-value:        {p_value:.4e}")
    
    if p_value < alpha:
        print(f"\nResult: Reject the null hypothesis (p < {alpha}).")
        print("There is a statistically significant difference between the models.")
        
        # 6. Post-Hoc Nemenyi Test
        print("\n--- Post-Hoc Nemenyi Test (Pairwise p-values) ---")
        nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_df.values)
        
        nemenyi_results.columns = models
        nemenyi_results.index = models
        
        print(nemenyi_results.round(4))
        
        print("\n--- Significant Pairwise Differences (p < 0.05) ---")
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                model_a = models[i]
                model_b = models[j]
                p_val = nemenyi_results.loc[model_a, model_b]
                
                if p_val < alpha:
                    better_model = model_a if avg_ranks[model_a] < avg_ranks[model_b] else model_b
                    worse_model = model_b if better_model == model_a else model_a
                    print(f"✅ {better_model} significantly outperforms {worse_model} (p = {p_val:.4f})")
    else:
        print(f"\nResult: Fail to reject the null hypothesis (p >= {alpha}).")
        print("There is NO statistically significant difference between the models.")
        nemenyi_results = None
        
    return pivot_df, avg_ranks, nemenyi_results


# Model variants grouped by forecasting head. The statistical tests are run
# separately per family so LSTM and MLP variants are never ranked together.
MODEL_FAMILIES = {
    'lstm': ['lstm_baseline', 'graph2vec_lstm', 'gcn_lstm', 'gat_lstm'],
    'mlp':  ['mlp_baseline', 'graph2vec_mlp', 'gcn_mlp', 'gat_mlp'],
}


def run_demsar_framework_by_family(data_input, metric='rmse', alpha=0.05, mode='total', seed_value=None, compare_ablations=False, model_family='lstm'):
    """
    Executes the Demšar statistical framework (Friedman + Nemenyi).
    compare_ablations: If True, treats ablate_z=True rows as separate models. 
                       If False, filters them out to only evaluate full models.
    model_family:      Which group of models to compare. 'lstm' restricts the test
                       to (lstm_baseline, graph2vec_lstm, gcn_lstm, gat_lstm); 'mlp'
                       restricts it to (mlp_baseline, graph2vec_mlp, gcn_mlp,
                       gat_mlp). Use 'all'/None to compare every variant together.
    """
    if mode == 'per_seed' and seed_value is None:
        raise ValueError("You must provide a 'seed_value' when using mode='per_seed'.")

    # 1. Load Data
    if isinstance(data_input, str):
        df = pd.read_csv(data_input)
    elif isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise ValueError("data_input must be a file path string or a pandas DataFrame")

    # ---------------------------------------------------------
    # Restrict to a single model family (LSTM or MLP) so the
    # statistical comparison only ranks like-for-like heads.
    # ---------------------------------------------------------
    if model_family not in (None, 'all'):
        if model_family not in MODEL_FAMILIES:
            raise ValueError(f"model_family must be one of {list(MODEL_FAMILIES)}, 'all', or None.")
        family_models = MODEL_FAMILIES[model_family]
        df = df[df['model_variant'].isin(family_models)]
        print(f"[*] MODEL FAMILY: '{model_family}' -> {family_models}")

    # ---------------------------------------------------------
    # NEW: Handle Ablations elegantly
    # ---------------------------------------------------------
    if compare_ablations:
        # Identify rows where ablate_z is True
        is_ablated = df['ablate_z'] == True
        # Rename them so the pivot table treats them as a brand new model
        df.loc[is_ablated, 'model_variant'] = df.loc[is_ablated, 'model_variant'] + '_ablated'
        print(f"[*] ABLATION MODE: Treating ablate_z=True as separate models.")
    else:
        # Filter OUT ablated runs. Keep only ablate_z=False OR baselines (which are NaN)
        df = df[(df['ablate_z'] == False) | (df['ablate_z'].isna())]
        print(f"[*] STANDARD MODE: Filtered out ablate_z=True runs.")

    # 2. Handle Modes (Filtering and Block Selection)
    if mode == 'per_seed':
        print(f"\n{'='*60}")
        print(f"--- Running Statistical Evaluation for {metric.upper()} | SEED: {seed_value} ---")
        print(f"{'='*60}")
        df = df[df['seed'] == seed_value]
        block_cols = ['product_id']
        pivot_agg = 'first'
    elif mode == 'per_product':
        # Conservative blocking: collapse seeds into one value per product so each
        # block is a genuinely independent dataset (N = n_products, e.g. 60), in
        # line with the canonical Demsar framework. This is the robustness check
        # against the liberal product x seed blocking used by mode='total'.
        print(f"\n{'='*60}")
        print(f"--- Running Statistical Evaluation for {metric.upper()} | PER-PRODUCT (mean over seeds) ---")
        print(f"{'='*60}")
        block_cols = ['product_id']
        pivot_agg = 'mean'
    else:
        print(f"\n{'='*60}")
        print(f"--- Running Statistical Evaluation for {metric.upper()} | TOTAL SEEDS ---")
        print(f"{'='*60}")
        block_cols = ['product_id', 'seed']
        pivot_agg = 'first'

    # 3. Pivot the data to get blocks
    pivot_df = df.pivot_table(
        index=block_cols,
        columns='model_variant',
        values=metric,
        aggfunc=pivot_agg
    )
    
    initial_rows = len(pivot_df)
    pivot_df = pivot_df.dropna()
    print(f"Loaded {initial_rows} blocks.")
    print(f"Retained {len(pivot_df)} complete blocks after dropping missing runs.\n")
    
    if len(pivot_df) < 2:
        print("Not enough complete blocks to run statistical tests.")
        return None, None, None

    models = pivot_df.columns.tolist()
    
    # 4. Calculate Average Ranks
    # Rank 1 = best. POCID / R2 are higher-is-better, error metrics lower-is-better.
    ranks = pivot_df.rank(axis=1, method='average', ascending=not _higher_is_better(metric))
    avg_ranks = ranks.mean().sort_values()
    
    print("--- Average Ranks (Lower is Better) ---")
    for model, rank in avg_ranks.items():
        print(f"{model:<25} : {rank:.3f}")
    print("\n")
    
    # 5. Omnibus Friedman Test
    data_arrays = [pivot_df[model].values for model in models]
    stat, p_value = stats.friedmanchisquare(*data_arrays)
    
    print("--- Friedman Test Results ---")
    print(f"Test Statistic: {stat:.4f}")
    print(f"P-value:        {p_value:.4e}")
    
    if p_value < alpha:
        print(f"\nResult: Reject the null hypothesis (p < {alpha}).")
        print("There is a statistically significant difference between the models.")
        
        # 6. Post-Hoc Nemenyi Test
        print("\n--- Post-Hoc Nemenyi Test (Pairwise p-values) ---")
        nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_df.values)
        
        nemenyi_results.columns = models
        nemenyi_results.index = models
        
        print(nemenyi_results.round(4))
        
        print("\n--- Significant Pairwise Differences (p < 0.05) ---")
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                model_a = models[i]
                model_b = models[j]
                p_val = nemenyi_results.loc[model_a, model_b]
                
                if p_val < alpha:
                    better_model = model_a if avg_ranks[model_a] < avg_ranks[model_b] else model_b
                    worse_model = model_b if better_model == model_a else model_a
                    print(f"✅ {better_model} significantly outperforms {worse_model} (p = {p_val:.4f})")
    else:
        print(f"\nResult: Fail to reject the null hypothesis (p >= {alpha}).")
        print("There is NO statistically significant difference between the models.")
        nemenyi_results = None
        
    return pivot_df, avg_ranks, nemenyi_results

# =============================================================================
# ROBUST STATISTICAL ANALYSIS (product-level dependence)
# -----------------------------------------------------------------------------
# The 'total' mode above treats every (product, seed) pair as an independent
# block (60 x 10 = 600). Seeds are repeated runs on the SAME dataset, so those
# blocks are correlated and the Friedman/Nemenyi p-values are too liberal.
# The two approaches below respect that structure:
#
#   A) product_level_analysis(...)          -> aggregate seeds per product (N = 60):
#      Friedman + Iman-Davenport + Kendall's W, Nemenyi, pairwise Wilcoxon
#      signed-rank tests with Holm-adjusted p-values, rank-biserial effect
#      sizes, and BCa bootstrap CIs over products.
#
#   B) hierarchical_bootstrap_analysis(...) -> cluster (product-level) paired
#      bootstrap over all 600 runs: products are resampled with replacement and
#      seeds stay nested inside their product (optionally resampled too).
#      Gives CIs and p-values for mean differences and for average ranks.
#
#   run_robust_analysis(...) runs both for every metric x family and saves CSVs.
# =============================================================================

ROBUST_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Robustness Results")


def _load_filtered(data_input, compare_ablations=False, model_family=None):
    """Load results and apply the same ablation / family filtering as the Demsar functions."""
    if isinstance(data_input, str):
        df = pd.read_csv(data_input)
    elif isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise ValueError("data_input must be a file path string or a pandas DataFrame")

    if model_family not in (None, 'all'):
        if model_family not in MODEL_FAMILIES:
            raise ValueError(f"model_family must be one of {list(MODEL_FAMILIES)}, 'all', or None.")
        df = df[df['model_variant'].isin(MODEL_FAMILIES[model_family])]

    if compare_ablations:
        df = df.copy()
        is_ablated = df['ablate_z'] == True
        df.loc[is_ablated, 'model_variant'] = df.loc[is_ablated, 'model_variant'] + '_ablated'
    else:
        df = df[(df['ablate_z'] == False) | (df['ablate_z'].isna())]
    return df


def _complete_cube(df, metric):
    """
    Build a dense array X[product, seed, model] keeping only products that have
    every model for every seed (balanced design). Returns (X, products, seeds, models).
    """
    wide = df.pivot_table(index=['product_id', 'seed'], columns='model_variant',
                          values=metric, aggfunc='first').dropna()
    models = wide.columns.tolist()
    seeds_per_prod = wide.groupby(level='product_id').size()
    n_seeds = seeds_per_prod.max()
    keep = seeds_per_prod[seeds_per_prod == n_seeds].index
    dropped = len(seeds_per_prod) - len(keep)
    if dropped:
        print(f"[!] Dropping {dropped} product(s) with incomplete seeds for the hierarchical bootstrap.")
    wide = wide.loc[keep].sort_index()
    products = wide.index.get_level_values('product_id').unique().tolist()
    seeds = sorted(wide.index.get_level_values('seed').unique().tolist())
    X = wide.values.reshape(len(products), n_seeds, len(models))
    return X, products, seeds, models


def _holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for k, idx in enumerate(order):
        running = max(running, (m - k) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def _pairs(models, reference=None):
    if reference is None:
        return list(combinations(models, 2))
    if reference not in models:
        raise ValueError(f"reference '{reference}' not among models {models}")
    return [(m, reference) for m in models if m != reference]


def _rank_biserial(d):
    """Matched-pairs rank-biserial correlation (Kerby, 2014). Positive => d > 0 dominates."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def _improvement(a, b, metric):
    """Per-block improvement of A over B, oriented so that POSITIVE = A is better."""
    return (a - b) if _higher_is_better(metric) else (b - a)


def product_level_analysis(data_input, metric='mae', model_family=None, compare_ablations=False,
                           reference=None, alpha=0.05, agg='mean', n_boot=10000, random_state=0):
    """
    APPROACH A - Product-level analysis (N = number of products, seeds aggregated).

    Each product is one independent block; its metric is the `agg` ('mean' or
    'median') over seeds. Reports:
      * Friedman + Iman-Davenport tests, Kendall's W (omnibus effect size),
        Nemenyi p-values and critical difference
      * Pairwise Wilcoxon signed-rank tests, Holm-adjusted within the family
      * Effect sizes: matched-pairs rank-biserial r, mean / median improvement,
        mean relative improvement (%), win / tie / loss counts over products
      * 95% BCa bootstrap CI (over products) for the mean improvement

    reference: if given (e.g. 'lstm_baseline'), only compares every model against
               it; otherwise all pairs. Improvement is oriented so that a POSITIVE
               value means model_a beats model_b, for both error and POCID metrics.

    Returns (omnibus: dict, pairwise: DataFrame, avg_ranks: Series, nemenyi: DataFrame)
    avg_ranks / nemenyi can be passed straight to plot_cd_diagram_per_family.
    """
    df = _load_filtered(data_input, compare_ablations, model_family)
    prod = df.pivot_table(index='product_id', columns='model_variant',
                          values=metric, aggfunc=agg).dropna()
    models = prod.columns.tolist()
    n, k = prod.shape
    hib = _higher_is_better(metric)

    print(f"\n{'='*70}")
    print(f"APPROACH A | PRODUCT-LEVEL | {metric.upper()} | family={model_family} | N={n} products, k={k}")
    print(f"{'='*70}")

    avg_ranks = prod.rank(axis=1, method='average', ascending=not hib).mean().sort_values()
    chi2, p_fr = stats.friedmanchisquare(*[prod[m].values for m in models])
    kendall_w = chi2 / (n * (k - 1))
    # Iman-Davenport F correction (recommended by Demsar, 2006)
    denom = n * (k - 1) - chi2
    ff = (n - 1) * chi2 / denom if denom > 0 else np.inf
    p_ff = stats.f.sf(ff, k - 1, (k - 1) * (n - 1))
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    omnibus = dict(metric=metric, model_family=model_family, n_products=n, k_models=k,
                   friedman_chi2=chi2, friedman_p=p_fr, iman_davenport_F=ff,
                   iman_davenport_p=p_ff, kendall_W=kendall_w, nemenyi_CD=cd)

    nemenyi = sp.posthoc_nemenyi_friedman(prod.values)
    nemenyi.index = nemenyi.columns = models

    print("Average ranks (1 = best):")
    for m, r in avg_ranks.items():
        print(f"  {m:<25} {r:.3f}")
    print(f"Friedman chi2={chi2:.3f}, p={p_fr:.3e} | Iman-Davenport F={ff:.3f}, p={p_ff:.3e} | "
          f"Kendall's W={kendall_w:.3f} | Nemenyi CD={cd:.3f}")

    rng = np.random.default_rng(random_state)
    rows = []
    for a, b in _pairs(models, reference):
        imp = _improvement(prod[a].values, prod[b].values, metric)
        base = np.abs(prod[b].values)
        rel = np.full_like(imp, np.nan, dtype=float)
        np.divide(imp * 100, base, out=rel, where=base > 0)
        if np.allclose(imp, 0):
            w_stat, p_w = np.nan, 1.0
        else:
            w_stat, p_w = stats.wilcoxon(imp, zero_method='wilcox', alternative='two-sided')
        try:
            ci = stats.bootstrap((imp,), np.mean, n_resamples=n_boot, method='BCa',
                                 confidence_level=1 - alpha, random_state=rng).confidence_interval
            ci_lo, ci_hi = ci.low, ci.high
        except Exception:
            ci_lo = ci_hi = np.nan
        rows.append(dict(model_a=a, model_b=b,
                         mean_a=prod[a].mean(), mean_b=prod[b].mean(),
                         mean_improvement=imp.mean(), ci_low=ci_lo, ci_high=ci_hi,
                         median_improvement=np.median(imp),
                         mean_rel_improvement_pct=np.nanmean(rel),
                         wins_a=int((imp > 0).sum()), ties=int((imp == 0).sum()),
                         losses_a=int((imp < 0).sum()),
                         wilcoxon_W=w_stat, p_value=p_w,
                         rank_biserial_r=_rank_biserial(imp),
                         nemenyi_p=nemenyi.loc[a, b]))
    pairwise = pd.DataFrame(rows)
    pairwise['p_holm'] = _holm(pairwise['p_value'].values)
    pairwise['significant'] = pairwise['p_holm'] < alpha
    pairwise.insert(0, 'metric', metric)
    pairwise.insert(1, 'model_family', model_family)

    print("\nPairwise Wilcoxon (positive improvement = model_a better; Holm-adjusted):")
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(pairwise[['model_a', 'model_b', 'mean_improvement', 'ci_low', 'ci_high',
                        'mean_rel_improvement_pct', 'wins_a', 'losses_a',
                        'rank_biserial_r', 'p_value', 'p_holm', 'significant']].round(4).to_string(index=False))
    return omnibus, pairwise, avg_ranks, nemenyi


def hierarchical_bootstrap_analysis(data_input, metric='mae', model_family=None, compare_ablations=False,
                                    reference=None, alpha=0.05, n_boot=10000, resample_seeds=False,
                                    random_state=0):
    """
    APPROACH B - Hierarchical (cluster) paired bootstrap over all product x seed runs.

    Uses every run, but the resampling unit is the PRODUCT: each bootstrap draw
    samples N products with replacement and carries all of that product's seeds
    along, paired across models so comparisons stay matched.
    resample_seeds=True additionally resamples seeds with replacement inside each
    drawn product (two-stage bootstrap). The default resamples only the top level,
    which is the recommended choice for nested data (Ren et al., 2010).

    For every pair (A, B): mean improvement of A over B (seed-averaged per product,
    then averaged over products), percentile CI, two-sided bootstrap p-value
    (Holm-adjusted), P(A better), and the difference in average rank with CI.
    For every model: average rank (ranked within each product x seed block, as in
    the 600-block CD diagrams) with a bootstrap CI and P(ranked best).

    Returns (pairwise: DataFrame, rank_summary: DataFrame)
    """
    df = _load_filtered(data_input, compare_ablations, model_family)
    X, products, seeds, models = _complete_cube(df, metric)
    P, S, M = X.shape
    hib = _higher_is_better(metric)

    print(f"\n{'='*70}")
    print(f"APPROACH B | HIERARCHICAL BOOTSTRAP | {metric.upper()} | family={model_family} | "
          f"{P} products x {S} seeds, B={n_boot}, resample_seeds={resample_seeds}")
    print(f"{'='*70}")

    # Ranks inside each (product, seed) block: 1 = best
    R = stats.rankdata(-X if hib else X, axis=2)

    pairs = _pairs(models, reference)
    idx = {m: i for i, m in enumerate(models)}
    ia = np.array([idx[a] for a, _ in pairs])
    ib = np.array([idx[b] for _, b in pairs])

    def stat(Xs, Rs):
        prod_means = Xs.mean(axis=1)                       # (P, M) per-product mean over seeds
        diff = prod_means[:, ia] - prod_means[:, ib]       # (P, n_pairs)
        imp = diff if hib else -diff
        return imp.mean(axis=0), Rs.mean(axis=(0, 1))

    obs_imp, obs_rank = stat(X, R)

    rng = np.random.default_rng(random_state)
    boot_imp = np.empty((n_boot, len(pairs)))
    boot_rank = np.empty((n_boot, M))
    for bi in range(n_boot):
        pi = rng.integers(0, P, P)
        if resample_seeds:
            si = rng.integers(0, S, (P, S))
            Xs, Rs = X[pi[:, None], si], R[pi[:, None], si]
        else:
            Xs, Rs = X[pi], R[pi]
        boot_imp[bi], boot_rank[bi] = stat(Xs, Rs)

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    # Two-sided bootstrap p-value: how often the bootstrap distribution crosses 0
    le0 = (boot_imp <= 0).sum(axis=0)
    ge0 = (boot_imp >= 0).sum(axis=0)
    p_boot = np.minimum(1.0, 2 * (np.minimum(le0, ge0) + 1) / (n_boot + 1))
    boot_rank_diff = boot_rank[:, ib] - boot_rank[:, ia]   # positive = A has the better (lower) rank

    pairwise = pd.DataFrame(dict(
        metric=metric, model_family=model_family,
        model_a=[a for a, _ in pairs], model_b=[b for _, b in pairs],
        mean_improvement=obs_imp,
        ci_low=np.percentile(boot_imp, lo_q, axis=0),
        ci_high=np.percentile(boot_imp, hi_q, axis=0),
        boot_se=boot_imp.std(axis=0, ddof=1),
        prob_a_better=(boot_imp > 0).mean(axis=0),
        rank_diff=obs_rank[ib] - obs_rank[ia],
        rank_diff_ci_low=np.percentile(boot_rank_diff, lo_q, axis=0),
        rank_diff_ci_high=np.percentile(boot_rank_diff, hi_q, axis=0),
        p_boot=p_boot,
    ))
    pairwise['p_holm'] = _holm(pairwise['p_boot'].values)
    pairwise['significant'] = pairwise['p_holm'] < alpha

    best = boot_rank.argmin(axis=1)
    rank_summary = pd.DataFrame(dict(
        metric=metric, model_family=model_family, model=models,
        avg_rank=obs_rank,
        rank_ci_low=np.percentile(boot_rank, lo_q, axis=0),
        rank_ci_high=np.percentile(boot_rank, hi_q, axis=0),
        prob_best=[(best == j).mean() for j in range(M)],
        mean_metric=X.mean(axis=(0, 1)),
    )).sort_values('avg_rank').reset_index(drop=True)

    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print("Average ranks with product-level bootstrap CIs (1 = best):")
        print(rank_summary[['model', 'avg_rank', 'rank_ci_low', 'rank_ci_high', 'prob_best']].round(4).to_string(index=False))
        print("\nPairwise (positive improvement = model_a better; Holm-adjusted):")
        print(pairwise[['model_a', 'model_b', 'mean_improvement', 'ci_low', 'ci_high',
                        'prob_a_better', 'rank_diff', 'p_boot', 'p_holm', 'significant']].round(4).to_string(index=False))
    return pairwise, rank_summary


def run_robust_analysis(data_input, metrics=('mae', 'pocid'), families=('lstm', 'mlp'),
                        graph_type='spearman', reference=None, alpha=0.05, n_boot=10000,
                        resample_seeds=False, compare_ablations=False, save=True, random_state=0):
    """
    Runs Approach A (product-level) and Approach B (hierarchical bootstrap) for
    every metric x family and, if save=True, writes four CSVs to ROBUST_RESULTS_DIR:
      <graph>_productlevel_omnibus.csv, <graph>_productlevel_pairwise.csv,
      <graph>_hierboot_pairwise.csv,    <graph>_hierboot_ranks.csv
    Returns a dict with the same four DataFrames.
    """
    omni, pw_a, pw_b, rk_b = [], [], [], []
    for metric in metrics:
        for fam in families:
            o, pa, _, _ = product_level_analysis(data_input, metric, fam, compare_ablations,
                                                 reference, alpha, n_boot=n_boot, random_state=random_state)
            pb, rb = hierarchical_bootstrap_analysis(data_input, metric, fam, compare_ablations,
                                                     reference, alpha, n_boot, resample_seeds, random_state)
            omni.append(o); pw_a.append(pa); pw_b.append(pb); rk_b.append(rb)

    out = dict(productlevel_omnibus=pd.DataFrame(omni),
               productlevel_pairwise=pd.concat(pw_a, ignore_index=True),
               hierboot_pairwise=pd.concat(pw_b, ignore_index=True),
               hierboot_ranks=pd.concat(rk_b, ignore_index=True))
    if save:
        os.makedirs(ROBUST_RESULTS_DIR, exist_ok=True)
        tag = graph_type.lower() + ("_ablations" if compare_ablations else "")
        for name, table in out.items():
            path = os.path.join(ROBUST_RESULTS_DIR, f"{tag}_{name}.csv")
            table.to_csv(path, index=False)
            print(f"Saved: {path}")
    return out
