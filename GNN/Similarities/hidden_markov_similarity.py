import warnings
import numpy as np
import pandas as pd
import networkx as nx
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def normal_fisher_distance(mu1, sig1, mu2, sig2, eps=1e-8):
    """
    Fisher-Rao distance between two univariate normal distributions.
    Protected against zero / tiny variances.
    """
    sig1 = max(float(sig1), eps)
    sig2 = max(float(sig2), eps)

    l1 = (mu1 - mu2) ** 2 + 2.0 * (sig1 - sig2) ** 2
    l2 = (mu1 - mu2) ** 2 + 2.0 * (sig1 + sig2) ** 2
    f = np.sqrt(max(l1 * l2, eps))

    num = max(f + (mu1 - mu2) ** 2 + 2.0 * (sig1**2 + sig2**2), eps)
    den = max(4.0 * sig1 * sig2, eps)

    return np.sqrt(2.0) * (np.log(num) - np.log(den))


def _get_stationary_distribution(transmat, tol=1e-12, max_iter=10000):
    """
    Safer stationary distribution via power iteration.
    Works better than eigen-decomposition for numerically noisy matrices.
    """
    transmat = np.asarray(transmat, dtype=float)
    n = transmat.shape[0]

    # Row-normalize just in case of tiny numerical drift
    row_sums = transmat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transmat = transmat / row_sums

    p = np.full(n, 1.0 / n, dtype=float)

    for _ in range(max_iter):
        p_new = p @ transmat
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new

    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0:
        return np.full(n, 1.0 / n, dtype=float)

    return p / s


def _gini_index(x):
    x = np.asarray(x, dtype=np.float64).flatten()

    if x.size == 0:
        return 0.0

    if np.amin(x) < 0:
        x = x - np.amin(x)

    x = x + 1e-12
    x = np.sort(x)

    n = x.shape[0]
    index = np.arange(1, n + 1)

    denom = n * np.sum(x)
    if denom <= 0:
        return 0.0

    return np.sum((2 * index - n - 1) * x) / denom


def hmm_similarity(hmm1, hmm2, fisher_eps=1e-8, similarity_eps=1e-2):
    """
    Similarity between two Gaussian HMMs using:
    - stationary distributions
    - pairwise emission similarity
    - concentration of the joint match matrix Q
    """
    n = hmm1.n_components
    m = hmm2.n_components

    stationary1 = _get_stationary_distribution(hmm1.transmat_)
    stationary2 = _get_stationary_distribution(hmm2.transmat_)

    mu1 = np.asarray(hmm1.means_, dtype=float).reshape(n, -1)
    mu2 = np.asarray(hmm2.means_, dtype=float).reshape(m, -1)
    cov1 = np.asarray(hmm1.covars_, dtype=float).reshape(n, -1)
    cov2 = np.asarray(hmm2.covars_, dtype=float).reshape(m, -1)

    if mu1.shape[1] != 1 or mu2.shape[1] != 1:
        raise ValueError("This implementation currently expects univariate emissions.")

    sig1 = np.sqrt(np.clip(cov1[:, 0], fisher_eps, None))
    sig2 = np.sqrt(np.clip(cov2[:, 0], fisher_eps, None))
    mu1 = mu1[:, 0]
    mu2 = mu2[:, 0]

    Se_matrix = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            d = normal_fisher_distance(mu1[i], sig1[i], mu2[j], sig2[j], eps=fisher_eps)
            Se_matrix[i, j] = 1.0 / (d + similarity_eps)

    Q = np.outer(stationary1, stationary2) * Se_matrix
    Q_sum = Q.sum()

    if (not np.isfinite(Q_sum)) or Q_sum <= 0:
        return 0.0

    Q = Q / Q_sum

    # Note: row length is m, column length is n
    gini_rows = np.mean([m * _gini_index(Q[i, :]) / max(1, m - 1) for i in range(n)])
    gini_cols = np.mean([n * _gini_index(Q[:, j]) / max(1, n - 1) for j in range(m)])

    sim = 0.5 * (gini_rows + gini_cols)

    if not np.isfinite(sim):
        return 0.0

    return float(np.clip(sim, 0.0, 1.0))


def _fit_best_hmm_1d(
    x,
    n_components=3,
    n_iter=200,
    random_seeds=(0, 1, 2, 3, 4),
    min_covar=1e-4,
    min_obs_per_state=10,
):
    """
    Fit the best univariate Gaussian HMM to one series.
    Tries multiple seeds, and if needed, fewer hidden states.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    T = len(x)

    if T < 3:
        raise ValueError("Series too short for HMM fitting.")

    # Reduce state count if series is too short or nearly constant
    series_std = float(np.std(x))
    max_states_by_length = max(1, T // min_obs_per_state)
    max_states = min(n_components, max_states_by_length if max_states_by_length > 0 else 1)

    if np.isclose(series_std, 0.0):
        max_states = 1

    max_states = max(1, max_states)

    best_model = None
    best_score = -np.inf
    best_info = None
    fallback_model = None
    fallback_score = -np.inf
    fallback_info = None

    for n_states in range(max_states, 0, -1):
        converged_candidates = []

        for seed in random_seeds:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=n_iter,
                random_state=seed,
                min_covar=min_covar,
            )

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model.fit(x)

                score = model.score(x)
                converged = bool(getattr(model.monitor_, "converged", False))

                info = {
                    "n_states_used": n_states,
                    "seed": seed,
                    "converged": converged,
                    "loglik": float(score),
                }

                if np.isfinite(score):
                    if score > fallback_score:
                        fallback_score = score
                        fallback_model = model
                        fallback_info = info

                    if converged:
                        converged_candidates.append((score, model, info))

            except Exception:
                continue

        if converged_candidates:
            converged_candidates.sort(key=lambda t: t[0], reverse=True)
            best_score, best_model, best_info = converged_candidates[0]
            break

    if best_model is not None:
        return best_model, best_info

    if fallback_model is not None:
        return fallback_model, fallback_info

    raise ValueError("HMM fitting failed for all seeds / state counts.")


def build_hmm_graph(
    df: pd.DataFrame,
    date_col: str,
    item_col: str,
    target_col: str,
    aggfunc: str = "sum",
    n_components: int = 3,
    similarity_threshold: float = 0.5,
    use_log1p: bool = False,
    train_end=None,
    n_iter: int = 200,
    random_seeds=(0, 1, 2, 3, 4),
    min_covar: float = 1e-4,
    min_obs_per_state: int = 10,
    verbose: bool = True,
    return_fit_info: bool = False,
):
    """
    Build an item-item graph using HMM-based similarity.

    Parameters
    ----------
    train_end : optional
        If provided, only data with date <= train_end is used to build the graph.
        Useful to avoid leakage in forecasting setups.
    return_fit_info : bool
        If True, also returns a DataFrame with per-item HMM fit diagnostics.

    Returns
    -------
    G : nx.Graph
    sim_df : pd.DataFrame
    df_scaled : pd.DataFrame
    fit_info_df : pd.DataFrame (optional)
    """
    df_work = df.copy()

    # Safer date handling when train_end is used
    if train_end is not None:
        df_work[date_col] = pd.to_datetime(df_work[date_col])
        train_end = pd.Timestamp(train_end)
        df_work = df_work[df_work[date_col] <= train_end]

    df_pivot = (
        df_work.pivot_table(
            index=date_col,
            columns=item_col,
            values=target_col,
            aggfunc=aggfunc,
        )
        .sort_index()
        .fillna(0)
    )

    if df_pivot.empty:
        raise ValueError("No data left after filtering / pivoting.")

    if use_log1p:
        df_pivot = np.log1p(df_pivot)

    scaler = StandardScaler(with_mean=True, with_std=True)
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_pivot),
        index=df_pivot.index,
        columns=df_pivot.columns,
    )

    item_ids = df_scaled.columns.tolist()
    X = df_scaled.T.to_numpy()
    n_items = len(item_ids)

    G = nx.Graph(name="HMM_Graph")
    G.add_nodes_from(item_ids)

    models = [None] * n_items
    fit_rows = []

    for i, item_id in enumerate(item_ids):
        x = X[i]

        try:
            model, info = _fit_best_hmm_1d(
                x=x,
                n_components=n_components,
                n_iter=n_iter,
                random_seeds=random_seeds,
                min_covar=min_covar,
                min_obs_per_state=min_obs_per_state,
            )
            models[i] = model

            G.nodes[item_id]["fit_success"] = True
            G.nodes[item_id]["n_states_used"] = info["n_states_used"]
            G.nodes[item_id]["converged"] = info["converged"]
            G.nodes[item_id]["loglik"] = info["loglik"]

            fit_rows.append({
                "item_id": item_id,
                "fit_success": True,
                **info,
            })

        except Exception as e:
            G.nodes[item_id]["fit_success"] = False
            G.nodes[item_id]["n_states_used"] = np.nan
            G.nodes[item_id]["converged"] = False
            G.nodes[item_id]["loglik"] = np.nan

            fit_rows.append({
                "item_id": item_id,
                "fit_success": False,
                "n_states_used": np.nan,
                "seed": np.nan,
                "converged": False,
                "loglik": np.nan,
                "error": str(e),
            })

            if verbose:
                print(f"[WARN] HMM fit failed for item {item_id}: {e}")

    sim_matrix = np.zeros((n_items, n_items), dtype=float)
    np.fill_diagonal(sim_matrix, 1.0)

    for i in range(n_items):
        for j in range(i + 1, n_items):
            if models[i] is None or models[j] is None:
                sim = 0.0
            else:
                try:
                    sim = hmm_similarity(models[i], models[j])
                except Exception as e:
                    sim = 0.0
                    if verbose:
                        print(f"[WARN] Similarity failed for ({item_ids[i]}, {item_ids[j]}): {e}")

            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

            if sim >= similarity_threshold:
                G.add_edge(item_ids[i], item_ids[j], weight=float(sim))
                if verbose:
                    print(
                        f"Added edge between {item_ids[i]} and {item_ids[j]} "
                        f"with HMM similarity: {sim:.4f}"
                    )

    sim_df = pd.DataFrame(sim_matrix, index=item_ids, columns=item_ids)
    fit_info_df = pd.DataFrame(fit_rows)

    if verbose:
        print("Number of nodes in the HMM graph:", G.number_of_nodes())
        print("Number of edges in the HMM graph:", G.number_of_edges())
        print("Successful HMM fits:", int(fit_info_df["fit_success"].sum()), "/", len(fit_info_df))

    if return_fit_info:
        return G, sim_df, df_scaled, fit_info_df

    return G, sim_df, df_scaled