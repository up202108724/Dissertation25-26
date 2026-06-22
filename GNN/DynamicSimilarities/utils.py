import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
import os
import seaborn as sns
import torch
from tslearn.metrics import cdist_dtw
import torch
from tslearn.metrics import dtw

DISTANCE_METRICS = {
    'euclidean', 'hamming', 'amplitude_offset', 'slope_consistency',
    'phase_invariance', 'dtw', 'cid', 'lorentzian', 'sbd', 'msm', 'edr', 'lcss',
    'manhattan', 'twed', 'erp', 'stid',
}
SIMILARITY_METRICS = {'pearson', 'spearman', 'kendall'}
CAUSAL_METRICS = {'granger'}

# Read catch22/24 feature names directly from pycatch22 so they always match
# the installed library version.  A short dummy series is enough to get names.
try:
    import pycatch22 as _pycatch22
    _catch24_names: list = list(_pycatch22.catch22_all(list(range(50)), catch24=True)["names"])
    _catch22_names: list = _catch24_names[:22]
except Exception:
    _catch22_names = [f'catch22_{i}' for i in range(22)]
    _catch24_names = _catch22_names + ['DN_Mean', 'DN_Spread_Std']

# Mirrors _MODE_FEATURE_LISTS in node_feature_builders — keep in sync.
_MODE_FEATURE_KEYS = {
    'raw':                ['raw'],
    'stats':              ['mean', 'std', 'min', 'max', 'first', 'last', 'slope', 'sum'],
    'catch22':            ['catch22'],
    'catch24':            ['catch24'],
    'catch24_minmax':     ['catch24', 'min', 'max'],
    'catch24_minmaxlast': ['catch24', 'min', 'max', 'last'],
}


def _node_feature_col_names(mode: str, n_feats: int) -> list:
    """Expand a node-feature mode string into ordered column names.

    Falls back to generic feat_i names for unknown modes or width mismatches.
    """
    keys = _MODE_FEATURE_KEYS.get(mode)
    if keys is None:
        return [f'feat_{i}' for i in range(n_feats)]
    cols = []
    for k in keys:
        if k == 'catch22':
            cols += _catch22_names
        elif k == 'catch24':
            cols += _catch24_names
        elif k == 'raw':
            n_other = sum(1 for kk in keys if kk != 'raw')
            cols += [f'raw_{j}' for j in range(n_feats - n_other)]
        else:
            cols.append(k)
    return cols if len(cols) == n_feats else [f'feat_{i}' for i in range(n_feats)]


def _save_node_features_csv(
    pyg_windows, product_id, store_id, metric, threshold, window_size, mode, script_dir
):
    """
    Dump node features from every PyG window graph to a CSV.
    Each row: window_idx, node_idx, <named feature columns>
    """
    out_dir = os.path.join(script_dir, "node_features")
    os.makedirs(out_dir, exist_ok=True)
    th_str = f"{threshold:.4f}" if threshold is not None else "None"
    fname = f"node_features_{product_id}_{store_id}_{metric}_th{th_str}_w{window_size}_{mode}.csv"
    out_path = os.path.join(out_dir, fname)

    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        n_feats = pyg_windows[0].x.shape[1] if len(pyg_windows) > 0 else 0
        feat_cols = _node_feature_col_names(mode, n_feats)
        writer.writerow(["window_idx", "node_idx"] + feat_cols)
        for w_idx, graph in enumerate(pyg_windows):
            x = graph.x.numpy() if hasattr(graph.x, "numpy") else graph.x
            for n_idx, row in enumerate(x):
                writer.writerow([w_idx, n_idx] + row.tolist())
    print(f"Node features saved to: {out_path}")


def infer_metric_type(metric, metric_type=None):
    if metric_type is not None:
        if metric_type not in {'distance', 'similarity', 'causal'}:
            raise ValueError("metric_type must be 'distance', 'similarity', or 'causal'")
        return metric_type
    if metric in DISTANCE_METRICS:
        return 'distance'
    if metric in SIMILARITY_METRICS:
        return 'similarity'
    if metric in CAUSAL_METRICS:
        return 'causal'
    raise ValueError(
        f"Metric {metric} not supported. "
        f"Distance metrics: {sorted(DISTANCE_METRICS)}; "
        f"similarity metrics: {sorted(SIMILARITY_METRICS)}; "
        f"causal metrics: {sorted(CAUSAL_METRICS)}"
    )


# Graph construction utils

def generate_exogenous_features(df, exog_cols, date_col='date', target_col='value', group_cols=None):
    """
    Generates specific calendar, cyclical, and holiday exogenous features for a DataFrame
    based on the provided `exog_cols` list.
    """
    df = df.copy()
    
    if group_cols is None:
        group_cols = [c for c in ['item_id', 'store_id'] if c in df.columns]
        if not group_cols:
            group_cols = None
    
    # -----------------------------------------------------------------------------
    # FEATURE BUILDER DICTIONARY
    # -----------------------------------------------------------------------------
    def _get_holidays():
        us_holidays = holidays.US()
        return us_holidays, pd.to_datetime(sorted(us_holidays.keys()))

    _holidays_cache = None
    def get_holiday_dates():
        nonlocal _holidays_cache
        if _holidays_cache is None:
            _holidays_cache = _get_holidays()
        return _holidays_cache

    builders = {
        # BASIC CALENDAR PARTS
        "day_of_week": lambda d: d[date_col].dt.dayofweek.astype(int),
        "day_of_month": lambda d: d[date_col].dt.day.astype(int),
        "month": lambda d: d[date_col].dt.month.astype(int),
        "moy": lambda d: (d[date_col].dt.month - 1).astype(int),
        "quarter": lambda d: d[date_col].dt.quarter.astype(int),
        "doy": lambda d: (d[date_col].dt.dayofyear - 1).astype(int),
        "week_of_year": lambda d: d[date_col].dt.isocalendar().week.astype(int),
        "year": lambda d: d[date_col].dt.year.astype(int),
        
        "is_weekend": lambda d: d[date_col].dt.dayofweek.isin([5, 6]).astype(int),
        "is_monday": lambda d: (d[date_col].dt.dayofweek == 0).astype(int),
        "is_friday": lambda d: (d[date_col].dt.dayofweek == 4).astype(int),

        "is_month_start": lambda d: d[date_col].dt.is_month_start.astype(int),
        "is_month_end": lambda d: d[date_col].dt.is_month_end.astype(int),
        "is_quarter_start": lambda d: d[date_col].dt.is_quarter_start.astype(int),
        "is_quarter_end": lambda d: d[date_col].dt.is_quarter_end.astype(int),
        "week_of_month": lambda d: ((d[date_col].dt.day - 1) // 7 + 1).astype(int),

            # CYCLICAL ENCODINGS — base harmonics
        "dow_sin": lambda d: np.sin(2 * np.pi * d[date_col].dt.dayofweek / 7),
        "dow_cos": lambda d: np.cos(2 * np.pi * d[date_col].dt.dayofweek / 7),

        "dom_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.day - 1) / 31.0),
        "dom_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.day - 1) / 31.0),

        "wom_sin": lambda d: np.sin(2 * np.pi * (((d[date_col].dt.day - 1) // 7)) / 5.0),
        "wom_cos": lambda d: np.cos(2 * np.pi * (((d[date_col].dt.day - 1) // 7)) / 5.0),

        "month_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.month - 1) / 12.0),
        "month_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.month - 1) / 12.0),

        "quarter_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.quarter - 1) / 4.0),
        "quarter_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.quarter - 1) / 4.0),

        "woy_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),
        "woy_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),

        "doy_sin": lambda d: np.sin(2 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),
        "doy_cos": lambda d: np.cos(2 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),

        # CYCLICAL ENCODINGS — 2nd harmonics
        "dow_sin2": lambda d: np.sin(4 * np.pi * d[date_col].dt.dayofweek / 7),
        "dow_cos2": lambda d: np.cos(4 * np.pi * d[date_col].dt.dayofweek / 7),

        "month_sin2": lambda d: np.sin(4 * np.pi * (d[date_col].dt.month - 1) / 12.0),
        "month_cos2": lambda d: np.cos(4 * np.pi * (d[date_col].dt.month - 1) / 12.0),

        "woy_sin2": lambda d: np.sin(4 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),
        "woy_cos2": lambda d: np.cos(4 * np.pi * (d[date_col].dt.isocalendar().week.astype(int) - 1) / 52.1775),

        "doy_sin2": lambda d: np.sin(4 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),
        "doy_cos2": lambda d: np.cos(4 * np.pi * (d[date_col].dt.dayofyear - 1) / 365.25),

        # EXACT HOLIDAYS
        "is_holiday": lambda d: d[date_col].isin(get_holiday_dates()[1]).astype(int),
        "is_christmas": lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 25)).astype(int),
        "is_thanksgiving": lambda d: d[date_col].apply(lambda x: 1 if get_holiday_dates()[0].get(x) == "Thanksgiving Day" else 0),
        "is_black_friday": lambda d: d[date_col].isin(
            pd.to_datetime([day for day, name in get_holiday_dates()[0].items() if name == "Thanksgiving Day"]) + pd.Timedelta(days=1)
        ).astype(int),
        "is_christmas_eve": lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 24)).astype(int),
        "is_new_year_eve": lambda d: ((d[date_col].dt.month == 12) & (d[date_col].dt.day == 31)).astype(int),

        # PROMOTIONS
        "promo_type_FRPG": lambda d: d.get("promo_type_FRPG", pd.Series(0, index=d.index)).astype(int),
        "promo_value_FRPG": lambda d: d.get("promo_value_FRPG", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_GAS": lambda d: d.get("promo_type_GAS", pd.Series(0, index=d.index)).astype(int),
        "promo_value_GAS": lambda d: d.get("promo_value_GAS", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_BOGO": lambda d: d.get("promo_type_BOGO", pd.Series(0, index=d.index)).astype(int),
        "promo_value_BOGO": lambda d: d.get("promo_value_BOGO", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_DISC": lambda d: d.get("promo_type_DISC", pd.Series(0, index=d.index)).astype(int),
        "promo_value_DISC": lambda d: d.get("promo_value_DISC", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_CIRC": lambda d: d.get("promo_type_CIRC", pd.Series(0, index=d.index)).astype(int),
        "promo_value_CIRC": lambda d: d.get("promo_value_CIRC", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_CIRE": lambda d: d.get("promo_type_CIRE", pd.Series(0, index=d.index)).astype(int),
        "promo_value_CIRE": lambda d: d.get("promo_value_CIRE", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_CLCP": lambda d: d.get("promo_type_CLCP", pd.Series(0, index=d.index)).astype(int),
        "promo_value_CLCP": lambda d: d.get("promo_value_CLCP", pd.Series(0.0, index=d.index)).astype(float),
        "promo_type_LFPE": lambda d: d.get("promo_type_LFPE", pd.Series(0, index=d.index)).astype(int),
        "promo_value_LFPE": lambda d: d.get("promo_value_LFPE", pd.Series(0.0, index=d.index)).astype(float),
        
        #Trend indicators could be added here as well, but we will handle them separately in the main function to avoid data leakage
        # TREND
        "time_idx": lambda d: (d[date_col] - d[date_col].min()).dt.days.astype(int),
        "time_idx_sq": lambda d: ((d[date_col] - d[date_col].min()).dt.days.astype(float) ** 2),
        
    }

    # Generate only the requested features
    for col in exog_cols:
        if col in builders:
            df[col] = builders[col](df)
        elif col.startswith("is_pre_holiday_") or col.startswith("is_post_holiday_"):
            parts = col.split("_")
            lag = int(parts[-1])
            is_pre = "pre" in parts
            
            _, holiday_dates = get_holiday_dates()
            
            # Efficiently compute proximity using sets and exact matches (avoiding loop row-by-row)
            df[col] = 0
            for h in holiday_dates:
                target_date = h - pd.Timedelta(days=lag) if is_pre else h + pd.Timedelta(days=lag)
                df.loc[df[date_col] == target_date, col] = 1
                
        elif col.startswith("lag_"):
            lag = int(col.split("_")[-1])
            if group_cols:
                df[col] = df.groupby(group_cols)[target_col].shift(lag).fillna(0)
            else:
                df[col] = df[target_col].shift(lag).fillna(0)
                
        elif col.startswith("rolling_mean_excl_"):
            window = int(col.split("_")[-1])
            if group_cols:
                df[col] = df.groupby(group_cols)[target_col].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()).fillna(0)
            else:
                df[col] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)
                
        elif col.startswith("rolling_mean_"):
            window = int(col.split("_")[-1])
            if group_cols:
                df[col] = df.groupby(group_cols)[target_col].transform(lambda x: x.rolling(window=window, min_periods=1).mean()).fillna(0)
            else:
                df[col] = df[target_col].rolling(window=window, min_periods=1).mean().fillna(0)

        elif col == "is_bridge_day":
            _, holiday_dates = get_holiday_dates()
            holiday_set = set(holiday_dates)
            
            df[col] = 0
            for idx in df.index:
                d = df.at[idx, date_col]
                prev_day = d - pd.Timedelta(days=1)
                next_day = d + pd.Timedelta(days=1)
                if ((prev_day in holiday_set and d.dayofweek == 4) or 
                    (next_day in holiday_set and d.dayofweek == 0)):
                    df.at[idx, col] = 1

        elif col in df.columns:
            # If the column exists in the dataset natively (e.g., store_id, cat_label) and isn't a builder key, just pass
            pass
                    
        else:
            print(f"Warning: Builder for feature '{col}' not found.")

    return df


def compute_similarities_1vsAll(target_ts, all_ts, metric='pearson', eps=1e-12):
    """
    Computes similarities between target_ts (1D) and all_ts (2D) using PyTorch.
    Optimized for 1-vs-all.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.tensor(target_ts, dtype=torch.float32, device=device).unsqueeze(0)
    X = torch.tensor(all_ts, dtype=torch.float32, device=device)
    
    if metric == 'pearson':
        target_mean = torch.mean(target, dim=1, keepdim=True)
        X_mean = torch.mean(X, dim=1, keepdim=True)
        
        target_centered = target - target_mean
        X_centered = X - X_mean
        
        cov = torch.sum(target_centered * X_centered, dim=1)
        target_var = torch.sqrt(torch.sum(target_centered**2, dim=1))
        X_var = torch.sqrt(torch.sum(X_centered**2, dim=1))
        
        sim = cov / (target_var * X_var + eps)
        return sim.cpu().numpy()
        
    elif metric == 'spearman':
        # Average-rank (fractional) ranking with proper tie handling, fully vectorised.
        # Ordinal ranking (argsort + scatter) is WRONG when the window contains ties:
        # tied values get sequential ranks in original-index order, which is identical
        # across every series, manufacturing spurious ~1.0 correlations. This is acute in
        # sparse early windows (many co-located zeros from .fillna(0)).
        def _avg_ranks(M):
            R, n = M.shape
            sorted_vals, order = torch.sort(M, dim=1)
            positions = torch.arange(n, device=device).unsqueeze(0).expand(R, n)
            # Mark group starts/ends among tied (equal) sorted values
            is_new = torch.ones(R, n, dtype=torch.bool, device=device)
            is_new[:, 1:] = sorted_vals[:, 1:] != sorted_vals[:, :-1]
            is_end = torch.ones(R, n, dtype=torch.bool, device=device)
            is_end[:, :-1] = sorted_vals[:, :-1] != sorted_vals[:, 1:]
            # First/last sorted position of each tie group, propagated to every member
            start_idx = torch.cummax(torch.where(is_new, positions, torch.zeros_like(positions)), dim=1)[0]
            end_pos = torch.where(is_end, positions, torch.full_like(positions, n))
            end_idx = torch.flip(torch.cummin(torch.flip(end_pos, [1]), dim=1)[0], [1])
            # Average ordinal rank over the group: ((start+1) + (end+1)) / 2
            avg_ord = (start_idx + end_idx).to(M.dtype) / 2.0 + 1.0
            ranks = torch.empty_like(M)
            ranks.scatter_(1, order, avg_ord)
            return ranks

        target_ranks = _avg_ranks(target)
        X_ranks = _avg_ranks(X)

        target_mean = torch.mean(target_ranks, dim=1, keepdim=True)
        X_mean = torch.mean(X_ranks, dim=1, keepdim=True)
        
        target_centered = target_ranks - target_mean
        X_centered = X_ranks - X_mean
        
        cov = torch.sum(target_centered * X_centered, dim=1)
        target_var = torch.sqrt(torch.sum(target_centered**2, dim=1))
        X_var = torch.sqrt(torch.sum(X_centered**2, dim=1))
        
        sim = cov / (target_var * X_var + eps)
        return sim.cpu().numpy()
        
    elif metric == 'kendall':
        seq_len = target.shape[1]
        if seq_len < 2:
            return torch.ones(X.shape[0], device=device).cpu().numpy()
            
        idx1, idx2 = torch.triu_indices(seq_len, seq_len, offset=1, device=device)
        
        target_diffs = target[:, idx1] - target[:, idx2]
        target_signs = torch.sign(target_diffs)
        
        X_diffs = X[:, idx1] - X[:, idx2]
        X_signs = torch.sign(X_diffs)
        
        S = torch.sum(target_signs * X_signs, dim=1)
        
        target_non_ties = torch.sum(target_signs**2, dim=1)
        X_non_ties = torch.sum(X_signs**2, dim=1)
        
        denom = torch.sqrt(target_non_ties * X_non_ties)
        
        sim = torch.where(denom == 0, torch.tensor(0.0, device=device), S / denom)
        return sim.cpu().numpy()
        
    else:
        raise ValueError(f"Metric {metric} not supported")



def edge_weight_from_metric(val, metric_type):
    """Map a raw metric value to a consistent edge weight where LARGER == MORE
    similar (a stronger edge), regardless of the metric's native direction.

    This is the single source of truth for edge weighting so every edge site
    (central star, within-star, second-degree) agrees, and so the downstream
    GCN (``edge_weight``) and GAT (``edge_attr`` prior) both interpret the
    weight the same way.

    similarity : ``max(0, val)``        — already "bigger = more similar".
    distance   : ``1 / (1 + val)``      — monotone-decreasing in distance,
                                          maps [0, ∞) → (0, 1] so a closer
                                          (smaller-distance) neighbour gets a
                                          larger weight, matching similarities.
    """
    if metric_type == 'distance':
        return 1.0 / (1.0 + float(val))
    return max(0.0, float(val))


def build_window_ego_graph(window_data, product_id, metric, metric_type, compute_func,
                           global_threshold, cat_labels=None,
                           enable_edges_within_star=True, enable_second_degree=False,
                           target_ts_override=None):
    """Build the ego-graph around ``product_id`` for a single window.

    This is the per-window construction extracted verbatim from
    ``neighbourhood_graph``'s Phase-2 loop, so that the *exact same* topology and
    edge weighting are produced both when building the training graphs and when
    rebuilding the graph step-by-step during recursive inference.

    Parameters
    ----------
    window_data        : DataFrame slice (items x window_size) in the metric's
                         working scale (raw for similarity, z-scored for
                         distance), indexed by item id and including
                         ``product_id``.
    global_threshold   : the resolved (training-period) threshold; reused for the
                         central star, within-star and second-degree edges.
    target_ts_override : optional 1-D array (length == window width) that
                         replaces the target row's series.  Used at inference so
                         the ego-graph is built from the model's own rolling
                         predictions instead of the unobserved future, while
                         every neighbour keeps its observed values.  When
                         ``None`` the target's own row of ``window_data`` is used
                         (training behaviour).

    Returns
    -------
    networkx.Graph — the ego-graph (target node first added).  ``start_date`` /
    ``end_date`` graph attributes are NOT set here; ``neighbourhood_graph`` adds
    them for the training list.
    """
    G = nx.Graph()

    cat = cat_labels.get(product_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
    G.add_node(product_id, cat_label=cat)

    if target_ts_override is not None:
        target_ts = np.asarray(target_ts_override, dtype=float)
    else:
        target_ts = window_data.loc[product_id].values
    all_ts = window_data.values
    item_ids = window_data.index.values

    vals = compute_func(target_ts, all_ts, metric=metric)

    active_items_mask = np.sum(np.abs(all_ts), axis=1) > 0
    valid_mask = (item_ids != product_id) & active_items_mask

    if np.sum(np.abs(target_ts)) == 0:
        valid_mask[:] = False

    valid_item_ids = item_ids[valid_mask]
    valid_vals = vals[valid_mask]
    valid_original_idxs = np.arange(len(item_ids))[valid_mask]

    if len(valid_vals) == 0:
        mask = np.array([], dtype=bool)
        current_threshold = 0
    else:
        if metric_type == 'distance':
            mask = valid_vals <= global_threshold
            current_threshold = global_threshold
        else:  # similarity
            mask = valid_vals >= global_threshold
            current_threshold = global_threshold

    selected_vals = valid_vals[mask]
    selected_ids = valid_item_ids[mask]
    selected_orig_idxs = valid_original_idxs[mask]

    # Garantir ordem determinística (fixar matriz de adjacências independentemente da query ou extração)
    sort_idx = np.argsort(selected_ids)
    selected_vals = selected_vals[sort_idx]
    selected_ids = selected_ids[sort_idx]
    selected_orig_idxs = selected_orig_idxs[sort_idx]

    neighbor_indices = []
    for orig_idx, val, other_id in zip(selected_orig_idxs, selected_vals, selected_ids):
        cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
        G.add_node(other_id, cat_label=cat_other)
        # Consistent weighting: bigger == more similar for every metric.
        # For similarity this is unchanged (selected vals are >= threshold > 0);
        # for distance this now matches the within-star / 2nd-degree edges.
        G.add_edge(product_id, other_id,
                   weight=edge_weight_from_metric(val, metric_type))
        neighbor_indices.append(orig_idx)

    if enable_edges_within_star and len(neighbor_indices) > 1:
        all_neighbors_ts = all_ts[neighbor_indices]
        for i, idx1 in enumerate(neighbor_indices):
            target_neighbor_ts = all_ts[idx1]
            vals_sub = compute_func(target_neighbor_ts, all_neighbors_ts, metric=metric)

            for j, (idx2, val_sub) in enumerate(zip(neighbor_indices, vals_sub)):
                if i < j:
                    passes = (val_sub <= current_threshold if metric_type == 'distance'
                              else val_sub >= current_threshold)
                    if passes:
                        edge_weight = edge_weight_from_metric(val_sub, metric_type)
                        G.add_edge(item_ids[idx1], item_ids[idx2], weight=edge_weight)

    if enable_second_degree and len(neighbor_indices) > 0:
        for idx1 in neighbor_indices:
            target_neighbor_ts = all_ts[idx1]
            vals_sub = compute_func(target_neighbor_ts, all_ts, metric=metric)

            for valid_idx, is_valid in enumerate(valid_mask):
                if is_valid and valid_idx != idx1:  # Must be active and not itself nor the central product
                    val_sub = vals_sub[valid_idx]
                    other_id = item_ids[valid_idx]

                    add_edge = (val_sub <= current_threshold if metric_type == 'distance'
                                else val_sub >= current_threshold)
                    if add_edge:
                        edge_weight = edge_weight_from_metric(val_sub, metric_type)

                    if add_edge:
                        if not G.has_node(other_id):
                            cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels is not None else "Unknown Category"
                            G.add_node(other_id, cat_label=cat_other)
                        if not G.has_edge(item_ids[idx1], other_id):
                            G.add_edge(item_ids[idx1], other_id, weight=edge_weight)

    return G


def neighbourhood_graph(product_id, df, metric, metric_type, window_size, compute_func,
                        threshold=None, percentile=None, step_size=1, cat_labels=None, plot_dir=None, residuals=False,
                        enable_edges_within_star=True, enable_second_degree=False, num_plots=None, train_end_idx=None):
    """
    Constructs a graph by iterating over sliding time windows of the time series data.
    Finds items within the specified metric thresholds or percentiles to product_id.
    
    Args:
        product_id: ID of the central product
        df: DataFrame with time series (rows=items, cols=time steps)
        metric: Metric string (e.g. 'euclidean', 'pearson')
        metric_type: 'distance' or 'similarity'
        window_size: Size of the sliding window
        compute_func: Function to compute 1-vs-All metric. Must take (target_ts, all_ts, metric)
        threshold: Strict cutoff threshold
        percentile: Top % of closest/most similar items
        step_size: Number of timesteps the window slides forward
        cat_labels: Dictionary mapping item IDs to category labels
        plot_dir: Save HTML plots in this directory
    """
    if threshold is None and percentile is None:
        raise ValueError("Must provide a threshold or percentile.")
        
    if product_id not in df.index:
        raise ValueError(f"Product ID {product_id} not found in DataFrame index.")
        
    time_steps = df.shape[1]
    
    # --- PHASE 1: Pre-calculate a global threshold across ALL windows ---
    global_threshold = threshold
    if threshold is None:
        all_valid_vals = []
        scan_end_limit = time_steps
        if train_end_idx is not None:
            scan_end_limit = min(scan_end_limit, train_end_idx)
            
        for start_idx in range(0, scan_end_limit - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_data = df.iloc[:, start_idx:end_idx]
            
            target_ts = window_data.loc[product_id].values
            if np.sum(np.abs(target_ts)) == 0:
                continue
                
            all_ts = window_data.values
            item_ids = window_data.index.values
            
            vals = compute_func(target_ts, all_ts, metric=metric)
            
            active_items_mask = np.sum(np.abs(all_ts), axis=1) > 0
            valid_mask = (item_ids != product_id) & active_items_mask
            all_valid_vals.extend(vals[valid_mask])
            
        all_valid_vals = np.array(all_valid_vals)
        all_valid_vals = all_valid_vals[np.isfinite(all_valid_vals)]
        
        if len(all_valid_vals) > 0:
            if percentile is not None:
                # Calculate exactly how many items correspond to the top-k%
                k = max(1, int(len(all_valid_vals) * (percentile / 100.0)))
                
                if metric_type == 'distance':
                    # Sort distances ascending (smaller is better). Slice top k, take the max.
                    top_k_vals = np.sort(all_valid_vals)[:k]
                    global_threshold = top_k_vals[-1]
                else: # similarity
                    # Sort similarities descending (larger is better). Slice top k, take the min.
                    top_k_vals = np.sort(all_valid_vals)[::-1][:k]
                    global_threshold = top_k_vals[-1]
        else:
            global_threshold = 0.0
            
    # --- PHASE 2: Build Graphs using the single global_threshold ---
    # The per-window construction is delegated to ``build_window_ego_graph`` so
    # the identical topology/weighting can be reused at recursive-inference time.
    graphs = []

    for start_idx in range(0, time_steps - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window_data = df.iloc[:, start_idx:end_idx]

        G = build_window_ego_graph(
            window_data, product_id, metric, metric_type, compute_func,
            global_threshold, cat_labels=cat_labels,
            enable_edges_within_star=enable_edges_within_star,
            enable_second_degree=enable_second_degree,
        )

        start_date = str(window_data.columns[0]).split(' ')[0].split('T')[0]
        end_date = str(window_data.columns[-1]).split(' ')[0].split('T')[0]

        G.graph['start_date'] = start_date
        G.graph['end_date'] = end_date

        graphs.append(G)

    return graphs, global_threshold
def compute_distances_1vsAll(target_ts, all_ts, metric='amplitude_offset', eps=1e-12, normalize_inputs=False):
    """
    Computes distances between target_ts (1D) and all_ts (2D) using PyTorch.
    Optimized for 1-vs-all.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.tensor(target_ts, dtype=torch.float32, device=device).unsqueeze(0)
    X = torch.tensor(all_ts, dtype=torch.float32, device=device)
    '''
    if normalize_inputs and metric in ['cid', 'dtw']:
        target_mean = torch.mean(target, dim=1, keepdim=True)
        target_std = torch.std(target, dim=1, keepdim=True) + eps
        target = (target - target_mean) / target_std
        
        X_mean = torch.mean(X, dim=1, keepdim=True)
        X_std = torch.std(X, dim=1, keepdim=True) + eps
        X = (X - X_mean) / X_std
    '''
    if metric == 'manhattan':
        # sum(|x - y|)
        dist = torch.sum(torch.abs(X - target), dim=1)
        return dist.cpu().numpy()
    
    elif metric == 'hamming':
        target_bin = (target > 0).float()
        X_bin = (X > 0).float()
        diffs = torch.abs(X_bin - target_bin)
        dist = torch.mean(diffs, dim=1)
        return dist.cpu().numpy()
    
    elif metric == 'amplitude_offset':
        target_mean = torch.mean(target, dim=1, keepdim=True)
        target_std = torch.std(target, dim=1, keepdim=True) + eps
        target_norm = (target - target_mean) / target_std
        
        X_mean = torch.mean(X, dim=1, keepdim=True)
        X_std = torch.std(X, dim=1, keepdim=True) + eps
        X_norm = (X - X_mean) / X_std
        
        dist = torch.cdist(target_norm, X_norm, p=2).squeeze(0)
        return dist.cpu().numpy()
        
    elif metric == 'slope_consistency':
        target_min = torch.min(target, dim=1, keepdim=True)[0]
        target_max = torch.max(target, dim=1, keepdim=True)[0]
        target_norm = (target - target_min) / (target_max - target_min + eps)
        
        X_min = torch.min(X, dim=1, keepdim=True)[0]
        X_max = torch.max(X, dim=1, keepdim=True)[0]
        X_norm = (X - X_min) / (X_max - X_min + eps)
        
        diffs = X_norm - target_norm
        dist = torch.var(diffs, dim=1, unbiased=False)
        return dist.cpu().numpy()
        
    elif metric == 'cid':
        ed_dist = torch.cdist(target, X, p=2).squeeze(0)
        
        # Complexity estimation
        ce_target = torch.sqrt(torch.sum(torch.diff(target, dim=1) ** 2, dim=1))
        ce_target_safe = torch.maximum(ce_target, torch.tensor(eps, device=device))
        
        ce_X = torch.sqrt(torch.sum(torch.diff(X, dim=1) ** 2, dim=1))
        ce_X_safe = torch.maximum(ce_X, torch.tensor(eps, device=device))
        
        ce_max = torch.maximum(ce_target_safe, ce_X_safe)
        ce_min = torch.minimum(ce_target_safe, ce_X_safe)
        cf_dist = ce_max / ce_min
        
        cid_dist = ed_dist * cf_dist
        return cid_dist.cpu().numpy()
        
    elif metric == 'dtw':
        # Use tslearn's optimized cdist_dtw for 1-vs-all vectorization 
        X_np = X.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # For a 15-point window, a Sakoe-Chiba radius of 2 or 3 (roughly ~10-20% of length) is optimal.
        # This constrains the pathological warping and drastically speeds up the calculation.
        dist = cdist_dtw(
            target_np.reshape(1, -1), 
            X_np, 
            global_constraint="sakoe_chiba", 
            sakoe_chiba_radius=2, 
            n_jobs=-1
        )
        
        # cdist_dtw returns a 2D distance matrix of shape (1, n_X), so we flatten it
        return dist.flatten()
        
    elif metric == 'phase_invariance':
        # Phase Invariance (Circular Shift) Euclidean Match
        # Find the minimum Euclidean distance across all possible circular shifts of the sequences
        X_np = X.cpu().numpy()
        target_np = target.cpu().numpy()
        
        n_X = X_np.shape[0]
        seq_len = X_np.shape[1]
        
        min_dists = np.full(n_X, np.inf)
        
        for shift in range(seq_len):
            shifted_X = np.roll(X_np, shift, axis=1)
            # Compute euclidean distance for this specific shift broadcasted across all rows
            current_dists = np.sqrt(np.sum((shifted_X - target_np)**2, axis=1))
            min_dists = np.minimum(min_dists, current_dists)
            
        return min_dists
        
    elif metric == 'lorentzian':
        # Lorentzian distance: sum(ln(1 + |x - y|))
        # Robust against outliers and noise
        diffs = torch.abs(X - target)
        dist = torch.sum(torch.log1p(diffs), dim=1)
        return dist.cpu().numpy()
    
    elif metric == 'twed':
        try:
            from sktime.distances import twe_distance
            X_np = X.cpu().numpy()
            target_np = target.cpu().numpy().flatten()
            # nu: stiffness, lmbda: penalty for deletion/insertion
            return np.array([twe_distance(target_np, x, nu=0.001, lmbda=1.0) for x in X_np])
        except ImportError as e:
            raise ValueError(f"metric='twed' failed to import: {e}")
        
    elif metric == 'erp':
        try:
            from sktime.distances import erp_distance
            X_np = X.cpu().numpy()
            target_np = target.cpu().numpy().flatten()
            # g is the gap value (usually 0.0 for Z-normalized data)
            return np.array([erp_distance(target_np, x, g=0.0) for x in X_np])
        except ImportError:
            raise ValueError("metric='erp' requires 'sktime'.")
        
    elif metric == 'stid':
        # Simplified STID: find min Euclidean distance across shifts after local scaling
        X_np = X.cpu().numpy()
        target_np = target.cpu().numpy()
        n_X, seq_len = X_np.shape
        min_dists = np.full(n_X, np.inf)
        
        # Search across possible shifts (w)
        for shift in range(-2, 3): # Local search window
            shifted_X = np.roll(X_np, shift, axis=1)
            # Optimal scaling alpha for each pair (simplified)
            # alpha = (X.T @ Y) / ||Y||^2
            dot_product = np.sum(shifted_X * target_np, axis=1)
            target_norm_sq = np.sum(target_np**2) + eps
            alpha = dot_product / target_norm_sq
            
            current_dists = np.linalg.norm(shifted_X - (alpha[:, None] * target_np), axis=1)
            min_dists = np.minimum(min_dists, current_dists)
            
        return min_dists
    
    elif metric == 'sbd':
        # Shape-Based Distance (SBD) using cross-correlation via FFT
        # SBD = 1 - max(NCC)
        X_np = X.cpu().numpy()
        target_np = target.cpu().numpy().flatten()
        
        n_X, seq_len = X_np.shape
        dists = np.zeros(n_X)
        
        target_norm = np.linalg.norm(target_np)
        target_norm = max(target_norm, eps)
        
        for i in range(n_X):
            x = X_np[i]
            x_norm = np.linalg.norm(x)
            x_norm = max(x_norm, eps)
            
            # Cross-correlation using scipy/numpy
            pad_len = 2 * seq_len - 1
            fft_x = np.fft.fft(x, n=pad_len)
            fft_y = np.fft.fft(target_np[::-1], n=pad_len)
            cc = np.real(np.fft.ifft(fft_x * fft_y))
            
            ncc = np.max(cc) / (target_norm * x_norm)
            # SBD falls in [0, 2]
            dists[i] = 1 - ncc
            
        return dists
        
    elif metric == 'msm':
        # Move-Split-Merge (MSM) is a metric elastic distance.
        try:
            from sktime.distances import msm_distance
            X_np = X.cpu().numpy()
            target_np = target.cpu().numpy().flatten()
            return np.array([msm_distance(target_np, x) for x in X_np])
        except ImportError:
            raise ValueError("metric='msm' requires 'sktime' to be installed. Run `pip install sktime`.")
            
    elif metric == 'edr' or metric == 'lcss':
        # Edit Distance on Real sequence (EDR) / Longest Common Subsequence (LCSS)
        X_np = X.cpu().numpy()
        target_np = target.cpu().numpy().flatten()
        
        if metric == 'lcss':
            try:
                from tslearn.metrics import lcss
                # Note: LCSS returns similarity ([0, 1]), so distance is 1 - similarity
                epsilon = 0.5 # A common threshold relative to scale; can be tuned
                return np.array([1 - lcss(target_np, x, eps=epsilon) for x in X_np])
            except ImportError:
                raise ValueError("metric='lcss' requires 'tslearn' to be installed.")
        else:
            try:
                from sktime.distances import edr_distance
                # EDR requires an epsilon threshold for what counts as a match
                epsilon = 0.5 
                return np.array([edr_distance(target_np, x, epsilon=epsilon) for x in X_np])
            except ImportError:
                raise ValueError("metric='edr' requires 'sktime' to be installed. Run `pip install sktime`.")
        
    else:
        raise ValueError(f"Metric {metric} not supported")
# Exogenous feature utils
def scale_exogenous_features(df, train_slice, categorical_cols=None, continuous_cols=None, binary_cols=None):
    df_scaled = df.copy()
    final_exog_cols = []
    
    cat_scaler = None
    cont_scaler = None
    
    # Scale categorical columns with OneHotEncoder
    if categorical_cols:
        cat_train = df.iloc[train_slice][categorical_cols].values
        cat_scaler = MinMaxScaler()  # Use MinMaxScaler to convert categories to a 0-1 range before OHE
        cat_scaler.fit(cat_train)
        
        cat_full_scaled = cat_scaler.transform(df[categorical_cols].values)
        cat_feature_names = cat_scaler.get_feature_names_out(categorical_cols)
        
        # Add new OHE columns and drop the originals
        df_cat = pd.DataFrame(cat_full_scaled, columns=cat_feature_names, index=df.index)
        df_scaled = pd.concat([df_scaled.drop(columns=categorical_cols), df_cat], axis=1)
        final_exog_cols.extend(cat_feature_names)
        
    # Scale continuous columns
    if continuous_cols:
        cont_train = df.iloc[train_slice][continuous_cols].values
        cont_scaler = MinMaxScaler()
        cont_scaler.fit(cont_train)
        
        cont_full_scaled = cont_scaler.transform(df[continuous_cols].values)
        df_scaled[continuous_cols] = cont_full_scaled
        final_exog_cols.extend(continuous_cols)
        
    # Keep binary columns as they are
    if binary_cols:
        final_exog_cols.extend(binary_cols)
        
    return df_scaled, final_exog_cols, cat_scaler, cont_scaler

def build_dynamic_exog_row(next_date, target_history, exog_cols,
                           lag_cols=None, rolling_cols=None, df_row=None):
    """
    Build one RAW exogenous row for next_date using current RAW target history.
    """
    row = {}

    if lag_cols:
        for col in lag_cols:
            lag = int(col.split('_')[1])
            row[col] = target_history[-lag] if len(target_history) >= lag else 0.0

    if rolling_cols:
        for col in rolling_cols:
            window = int(col.split('_')[2])
            if len(target_history) >= window:
                row[col] = float(np.mean(target_history[-window:]))
            else:
                row[col] = float(np.mean(target_history)) if len(target_history) > 0 else 0.0

    if df_row is not None:
        for col in exog_cols:
            if col not in row and col in df_row.index:
                row[col] = df_row[col]

    return pd.DataFrame([[row.get(col, 0.0) for col in exog_cols]], columns=exog_cols)



def transform_exog_row(row_df, categorical_cols=None, continuous_cols=None, binary_cols=None,
                       cat_scaler=None, cont_scaler=None, final_exog_cols=None):
    parts = []

    if categorical_cols:
        # Force column order
        cat_arr = cat_scaler.transform(row_df[categorical_cols].values)
        cat_names = cat_scaler.get_feature_names_out(categorical_cols)
        parts.append(pd.DataFrame(cat_arr, columns=cat_names, index=row_df.index))

    if continuous_cols:
        # Force column order precisely as fitted
        cont_arr = cont_scaler.transform(row_df[continuous_cols].values)
        parts.append(pd.DataFrame(cont_arr, columns=continuous_cols, index=row_df.index))

    if binary_cols:
        # Force column order
        parts.append(row_df[binary_cols].copy())

    out = pd.concat(parts, axis=1)

    if final_exog_cols is not None:
        # Reindex to ensure strictly ordered columns matching the training tensors
        out = out.reindex(columns=final_exog_cols, fill_value=0.0)

    return out




def compute_granger_graph(
    product_id,
    df_wide_train,
    max_lag=7,
    p_threshold=0.05,
    cat_labels=None,
):
    """
    Build a static undirected graph whose edges encode Granger-causal relationships.

    Run on the TRAINING period only — zero look-ahead leakage. First-differences
    each series before testing to satisfy the stationarity requirement. An edge
    between product_id and other_id is created if the F-test p-value is below
    p_threshold in either causal direction; edge weight = 1 - min_p (causality
    confidence in [0, 1]).

    The returned graph has fixed topology. Pass it replicated N times to
    build_pyg_graphs_from_nx_windows so the GCN sees the same causal structure
    at every window while node features refresh with each sliding window.
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    import warnings

    G = nx.Graph()
    cat = cat_labels.get(product_id, "Unknown Category") if cat_labels else "Unknown Category"
    G.add_node(product_id, cat_label=cat)

    if product_id not in df_wide_train.index:
        return G

    target_diff = np.diff(df_wide_train.loc[product_id].values.astype(float))
    if np.std(target_diff) < 1e-8:
        print(f"[Granger] product={product_id}: target is flat — no edges added")
        return G

    n_added = 0
    for other_id in df_wide_train.index:
        if other_id == product_id:
            continue

        other_diff = np.diff(df_wide_train.loc[other_id].values.astype(float))
        if np.std(other_diff) < 1e-8:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Does other_id Granger-cause product_id?
                r1 = grangercausalitytests(
                    np.column_stack([target_diff, other_diff]),
                    maxlag=max_lag, verbose=False,
                )
                p1 = min(r1[lag][0]['ssr_ftest'][1] for lag in r1)

                # Does product_id Granger-cause other_id?
                r2 = grangercausalitytests(
                    np.column_stack([other_diff, target_diff]),
                    maxlag=max_lag, verbose=False,
                )
                p2 = min(r2[lag][0]['ssr_ftest'][1] for lag in r2)

                min_p = min(p1, p2)
                if min_p < p_threshold:
                    cat_other = cat_labels.get(other_id, "Unknown Category") if cat_labels else "Unknown Category"
                    G.add_node(other_id, cat_label=cat_other)
                    G.add_edge(product_id, other_id, weight=float(1.0 - min_p))
                    n_added += 1

        except Exception:
            continue

    print(f"[Granger] product={product_id}: {n_added} causal neighbours "
          f"(p<{p_threshold}, max_lag={max_lag})")
    return G


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def compute_metrics(y_test, y_pred):
    
    def POCID(y_test, y_pred):
        diff_original = y_test[1:] - y_test[:-1]
        diff_pred = y_pred[1:] - y_pred[:-1]
        is_positive = (diff_original * diff_pred) > 0
        return is_positive.sum() / len(is_positive) if len(is_positive) > 0 else 0.0

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    bias = np.mean(y_pred - y_test)
    score = 0.5 * rmse + 0.25 * mae + 0.25 * abs(bias)
    pocid = POCID(y_test, y_pred)
    return rmse, mae, bias, score, pocid


