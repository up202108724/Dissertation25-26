

import torch
import numpy as np
import networkx as nx
import holidays
import pandas as pd

def compute_similarities_allvsall(all_ts, metric='pearson', eps=1e-12):
    """
    Computes pairwise similarities between all time series in all_ts (2D) using PyTorch.
    Returns an (N, N) symmetric similarity matrix.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(all_ts, dtype=torch.float32, device=device)
    
    if metric == 'pearson':
        X_mean = torch.mean(X, dim=1, keepdim=True)
        X_centered = X - X_mean
        
        # Covariance matrix (N, N) via matrix multiplication
        cov = torch.mm(X_centered, X_centered.T)
        
        # Standard deviations
        X_var = torch.sum(X_centered**2, dim=1)
        X_std = torch.sqrt(X_var)
        
        # Denominator matrix (N, N) via outer product
        denom = torch.ger(X_std, X_std) 
        
        sim = cov / (denom + eps)
        return sim.cpu().numpy()
        
    elif metric == 'spearman':
        # Compute ranks across dim 1
        _, X_indices = torch.sort(X, dim=1)
        X_ranks = torch.empty_like(X)
        arange = torch.arange(1, X.shape[1] + 1, dtype=torch.float32, device=device).unsqueeze(0).expand_as(X)
        X_ranks.scatter_(1, X_indices, arange)
        
        # Pearson correlation on the ranks
        X_mean = torch.mean(X_ranks, dim=1, keepdim=True)
        X_centered = X_ranks - X_mean
        
        cov = torch.mm(X_centered, X_centered.T)
        X_var = torch.sum(X_centered**2, dim=1)
        X_std = torch.sqrt(X_var)
        
        denom = torch.ger(X_std, X_std)
        
        sim = cov / (denom + eps)
        return sim.cpu().numpy()
        
    elif metric == 'kendall':
        N, seq_len = X.shape
        if seq_len < 2:
            return torch.ones((N, N), device=device).cpu().numpy()
            
        idx1, idx2 = torch.triu_indices(seq_len, seq_len, offset=1, device=device)
        
        # Differences and signs for all pairs of time points
        X_diffs = X[:, idx1] - X[:, idx2]
        X_signs = torch.sign(X_diffs)
        
        # S matrix (N, N): dot product of all sign vectors
        S = torch.mm(X_signs, X_signs.T)
        
        # Denominators based on non-ties
        X_non_ties = torch.sum(X_signs**2, dim=1)  # Shape: (N,)
        denom = torch.sqrt(torch.ger(X_non_ties, X_non_ties))  # Shape: (N, N)
        
        sim = torch.where(denom == 0, torch.tensor(0.0, device=device), S / denom)
        return sim.cpu().numpy()
        
    else:
        raise ValueError(f"Metric {metric} not supported")
    
    

def compute_distances_allvsall(all_ts, metric='amplitude_offset', eps=1e-12):
    """
    Computes the pairwise distance matrix between all time series in all_ts (2D) using PyTorch.
    Returns an (N, N) symmetric distance matrix.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(all_ts, dtype=torch.float32, device=device)
    
    if metric == 'manhattan':
        # p=1 computes the Manhattan distance pairwise
        dist = torch.cdist(X, X, p=1)
        return dist.cpu().numpy()
    
    elif metric == 'hamming':
        X_bin = (X > 0).float()
        # Broadcast to (N, N, L) and take the mean across the sequence length
        diffs = torch.abs(X_bin.unsqueeze(1) - X_bin.unsqueeze(0))
        dist = torch.mean(diffs, dim=2)
        return dist.cpu().numpy()
    
    elif metric == 'amplitude_offset':
        X_mean = torch.mean(X, dim=1, keepdim=True)
        X_std = torch.std(X, dim=1, keepdim=True) + eps
        X_norm = (X - X_mean) / X_std
        
        # Euclidean distance on Z-normalized data
        dist = torch.cdist(X_norm, X_norm, p=2)
        return dist.cpu().numpy()
        
    elif metric == 'slope_consistency':
        X_min = torch.min(X, dim=1, keepdim=True)[0]
        X_max = torch.max(X, dim=1, keepdim=True)[0]
        X_norm = (X - X_min) / (X_max - X_min + eps)
        
        diffs = X_norm.unsqueeze(1) - X_norm.unsqueeze(0)
        dist = torch.var(diffs, dim=2, unbiased=False)
        return dist.cpu().numpy()
        
    elif metric == 'cid':
        ed_dist = torch.cdist(X, X, p=2)
        
        # Complexity estimation
        ce_X = torch.sqrt(torch.sum(torch.diff(X, dim=1) ** 2, dim=1))
        ce_X_safe = torch.maximum(ce_X, torch.tensor(eps, device=device))
        
        # Cross-compare complexities
        ce_max = torch.maximum(ce_X_safe.unsqueeze(1), ce_X_safe.unsqueeze(0))
        ce_min = torch.minimum(ce_X_safe.unsqueeze(1), ce_X_safe.unsqueeze(0))
        cf_dist = ce_max / ce_min
        
        cid_dist = ed_dist * cf_dist
        return cid_dist.cpu().numpy()
        
    elif metric == 'dtw':
        try:
            from tslearn.metrics import cdist_dtw
            X_np = X.cpu().numpy()
            # cdist_dtw natively returns an (N, N) matrix
            dist = cdist_dtw(
                X_np, 
                X_np, 
                global_constraint="sakoe_chiba", 
                sakoe_chiba_radius=2, 
                n_jobs=-1
            )
            return dist
        except ImportError:
            raise ValueError("metric='dtw' requires 'tslearn' to be installed.")
        
    elif metric == 'phase_invariance':
        from scipy.spatial.distance import cdist
        X_np = X.cpu().numpy()
        N, seq_len = X_np.shape
        min_dists = np.full((N, N), np.inf)
        
        for shift in range(seq_len):
            shifted_X = np.roll(X_np, shift, axis=1)
            current_dists = cdist(X_np, shifted_X, metric='euclidean')
            min_dists = np.minimum(min_dists, current_dists)
            
        return min_dists
        
    elif metric == 'lorentzian':
        # Broadcasting to (N, N, L)
        diffs = torch.abs(X.unsqueeze(1) - X.unsqueeze(0))
        dist = torch.sum(torch.log1p(diffs), dim=2)
        return dist.cpu().numpy()
        
    elif metric in ['twed', 'erp', 'msm', 'edr']:
        # These are computationally heavy sktime metrics. 
        # We manually build the symmetric matrix to cut calculations by 50%.
        X_np = X.cpu().numpy()
        N = X_np.shape[0]
        dist_mat = np.zeros((N, N))
        
        if metric == 'twed':
            from sktime.distances import twe_distance as sk_dist
            kwargs = {'nu': 0.001, 'lmbda': 1.0}
        elif metric == 'erp':
            from sktime.distances import erp_distance as sk_dist
            kwargs = {'g': 0.0}
        elif metric == 'msm':
            from sktime.distances import msm_distance as sk_dist
            kwargs = {}
        elif metric == 'edr':
            from sktime.distances import edr_distance as sk_dist
            kwargs = {'epsilon': 0.5}

        for i in range(N):
            for j in range(i + 1, N): # Upper triangle only
                d = sk_dist(X_np[i], X_np[j], **kwargs)
                dist_mat[i, j] = dist_mat[j, i] = d
                
        return dist_mat
        
    elif metric == 'stid':
        X_np = X.cpu().numpy()
        N, seq_len = X_np.shape
        min_dists = np.full((N, N), np.inf)
        X_norm_sq = np.sum(X_np**2, axis=1) + eps
        
        for shift in range(-2, 3):
            shifted_X = np.roll(X_np, shift, axis=1)
            # Dot product of all pairs (N, N)
            dot_product = np.dot(shifted_X, X_np.T)
            
            # alpha is optimal scaling for each pair i, j
            alpha = dot_product / X_norm_sq[None, :] 
            
            # Vectorized distance using algebraic expansion: (A - alpha*B)^2 = A^2 - 2*alpha*(A.B) + alpha^2*B^2
            shifted_X_sq = np.sum(shifted_X**2, axis=1)[:, None]
            dist_sq = shifted_X_sq - (2 * alpha * dot_product) + (alpha**2 * X_norm_sq[None, :])
            
            # Clip to 0 to avoid tiny negative values from float inaccuracies before sqrt
            current_dists = np.sqrt(np.maximum(dist_sq, 0))
            min_dists = np.minimum(min_dists, current_dists)
            
        return min_dists
    
    elif metric == 'sbd':
        X_np = X.cpu().numpy()
        N, seq_len = X_np.shape
        dists = np.zeros((N, N))
        
        X_norms = np.linalg.norm(X_np, axis=1)
        X_norms = np.maximum(X_norms, eps)
        
        pad_len = 2 * seq_len - 1
        fft_X = np.fft.fft(X_np, n=pad_len, axis=1)
        fft_X_rev = np.fft.fft(X_np[:, ::-1], n=pad_len, axis=1)
        
        for i in range(N):
            # Compute cross-correlation of X[i] against all sequences efficiently
            cc_matrix = np.real(np.fft.ifft(fft_X[i:i+1] * fft_X_rev, axis=1))
            ncc = np.max(cc_matrix, axis=1) / (X_norms[i] * X_norms)
            dists[i, :] = 1 - ncc
            
        # Ensure exact symmetry and zeroes on the diagonal
        np.fill_diagonal(dists, 0)
        return (dists + dists.T) / 2
            
    elif metric == 'lcss':
        try:
            from tslearn.metrics import cdist_lcss
            X_np = X.cpu().numpy()
            sim_mat = cdist_lcss(X_np, X_np, eps=0.5)
            # lcss returns similarities in [0, 1], so we invert it for distances
            return 1 - sim_mat
        except ImportError:
            raise ValueError("metric='lcss' requires 'tslearn' to be installed.")
        
    else:
        raise ValueError(f"Metric {metric} not supported")
    
    
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



# =============================================================================
# Clustering: re-build the 0.6 Spearman graph and take connected components
# =============================================================================
def build_similarity_clusters(df, metric, threshold,date_col, target_col,
                              min_cluster_size):
    """Return (clusters, df_wide, cat_labels) where ``clusters`` is a list of
    item_id lists -- one per connected component of size >= min_cluster_size.

    Construction is identical to graph_global_analysis.py: an (item_id x date)
    pivot, an all-vs-all similarity matrix, thresholded into an undirected graph.
    """
    df_wide = (
        df.pivot_table(index='item_id', columns=date_col, values=target_col, aggfunc='sum')
        .fillna(0)
    )
    cat_labels = (
        df.drop_duplicates('item_id').set_index('item_id')['cat_label'].to_dict()
        if 'cat_label' in df.columns else {}
    )

    item_ids = df_wide.index.tolist()
    all_ts = df_wide.values.astype(np.float32)            # (N, T)

    print(f"Computing {metric} similarity for {len(item_ids)} x {len(item_ids)} pairs...")
    sim_matrix = compute_similarities_allvsall(all_ts, metric=metric)

    # Threshold -> undirected graph (exclude self-loops via upper triangle)
    mask = np.triu(sim_matrix >= threshold, k=1)
    rows, cols = np.where(mask)

    G = nx.Graph()
    G.add_nodes_from(item_ids)
    for i, j in zip(rows, cols):
        G.add_edge(item_ids[i], item_ids[j], weight=float(sim_matrix[i, j]))

    # Connected components -> clusters. Keep only those big enough for a
    # multivariate model; sort each by item_id for deterministic channel order.
    clusters = [
        sorted(comp)
        for comp in nx.connected_components(G)
        if len(comp) >= min_cluster_size
    ]
    clusters.sort(key=lambda c: (-len(c), c[0]))          # biggest first

    n_connected = sum(1 for _, d in G.degree() if d > 0)
    print(f"Threshold = {threshold} | metric = {metric}")
    print(f"  Connected products : {n_connected}")
    print(f"  Edges              : {G.number_of_edges()}")
    print(f"  Clusters (>= {min_cluster_size}) : {len(clusters)} "
          f"covering {sum(len(c) for c in clusters)} products")
    return clusters, df_wide, cat_labels