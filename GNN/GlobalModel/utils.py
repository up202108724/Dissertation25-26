

import torch
import numpy as np
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