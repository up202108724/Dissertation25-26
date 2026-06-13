from __future__ import annotations

import numpy as np
import pycatch22


_CATCH22_WIDTH = 22
_CATCH24_WIDTH = 24          # 22 catch22 + DN_Mean + DN_Spread_Std


# ── catch block helper ─────────────────────────────────────────────────────
def _compute_catch_block(window_values: np.ndarray, catch24: bool) -> np.ndarray:
    """(n_nodes, 22|24) catch22/24 features; crashes on degenerate series → 0."""
    n_nodes = window_values.shape[0]
    width = _CATCH24_WIDTH if catch24 else _CATCH22_WIDTH
    feats = np.zeros((n_nodes, width), dtype=np.float32)
    for i in range(n_nodes):
        ts = np.asarray(window_values[i], dtype=np.float64).tolist()
        try:
            vals = pycatch22.catch22_all(ts, catch24=catch24)["values"]
            feats[i] = np.asarray(vals, dtype=np.float32)
        except Exception:
            feats[i] = 0.0
    return feats


# ── feature registry ───────────────────────────────────────────────────────
# Scalar builders return (n_nodes,); block builders return (n_nodes, k).
# Add any new feature here — everything else derives from this dict.
_SCALAR_BUILDERS: dict[str, callable] = {
    'mean':          lambda w: w.mean(axis=1),
    'std':           lambda w: w.std(axis=1),
    'min':           lambda w: w.min(axis=1),
    'max':           lambda w: w.max(axis=1),
    'first':         lambda w: w[:, 0],
    'last':          lambda w: w[:, -1],
    'slope':         lambda w: (w[:, -1] - w[:, 0]) / max(w.shape[1] - 1, 1),
    'sum':           lambda w: w.sum(axis=1),
    'zero_fraction': lambda w: (w == 0).mean(axis=1),
}

_BLOCK_BUILDERS: dict[str, callable] = {
    'raw':     lambda w: w.copy(),
    'catch22': lambda w: _compute_catch_block(w, catch24=False),
    'catch24': lambda w: _compute_catch_block(w, catch24=True),
}

# 'raw' width is variable (= window_size); pass window_size to node_feature_width.
_FEATURE_WIDTHS: dict[str, int | None] = {
    **{k: 1 for k in _SCALAR_BUILDERS},
    'raw':     None,   # sentinel — resolved at runtime via window_size argument
    'catch22': _CATCH22_WIDTH,
    'catch24': _CATCH24_WIDTH,
}

ALL_NODE_FEATURES: list[str] = list(_SCALAR_BUILDERS) + list(_BLOCK_BUILDERS)


# ── mode presets ───────────────────────────────────────────────────────────
_MODE_FEATURE_LISTS: dict[str, list[str]] = {
    'raw':     ['raw'],
    'stats':   ['mean', 'std', 'min', 'max', 'first', 'last', 'slope', 'sum'],
    'catch22': ['catch22'],
    'catch24': ['catch24'],
}

# Extra scale/level features appended to catch22/catch24 when extra_stats=True.
# Edit this list to add/remove extra stats — widths stay in sync automatically.
_EXTRA_STAT_NAMES: list[str] = ['min', 'max', 'last', 'zero_fraction']

# Kept for callers that import the constant directly (e.g. gcn_gat_assmble.py).
_EXTRA_STATS_WIDTH: int = sum(_FEATURE_WIDTHS[n] for n in _EXTRA_STAT_NAMES)


# ── public API ─────────────────────────────────────────────────────────────
def generate_node_features(
    window_values: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """
    Build a node-feature matrix by selecting and concatenating features from
    the registry in the given order.

    window_values : (n_nodes, window_size) raw values per node
    feature_names : ordered list of feature keys; any combination of
                    scalar keys — 'mean', 'std', 'min', 'max', 'first',
                    'last', 'slope', 'sum', 'zero_fraction' — and block
                    keys — 'catch22' (22-d), 'catch24' (24-d).

    Returns : (n_nodes, W) float32 matrix, NaN/Inf replaced with 0.
    """
    if window_values.ndim != 2:
        raise ValueError("window_values must be 2D (n_nodes, window_size)")

    parts: list[np.ndarray] = []
    for name in feature_names:
        if name in _SCALAR_BUILDERS:
            col = _SCALAR_BUILDERS[name](window_values).astype(np.float32)
            parts.append(col[:, np.newaxis])
        elif name in _BLOCK_BUILDERS:
            parts.append(_BLOCK_BUILDERS[name](window_values).astype(np.float32))
        else:
            raise ValueError(
                f"Unknown node feature {name!r}. Available: {ALL_NODE_FEATURES}"
            )

    feats = (
        np.concatenate(parts, axis=1) if parts
        else np.empty((window_values.shape[0], 0), dtype=np.float32)
    )
    np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def node_feature_width(feature_names: list[str], window_size: int = 0) -> int:
    """
    Total feature dimension for a given feature list.

    window_size : required only when 'raw' is in feature_names (its width
                  equals the window length and cannot be inferred statically).
    """
    total = 0
    for n in feature_names:
        w = _FEATURE_WIDTHS[n]
        total += window_size if w is None else w
    return total


def window_node_features(
    window_values: np.ndarray,
    mode: str = 'stats',
    extra_stats: bool = True,
) -> np.ndarray:
    """
    Preset wrapper around generate_node_features.

    mode        : 'raw' | 'stats' | 'catch22' | 'catch24'
    extra_stats : catch22/catch24 only — appends _EXTRA_STAT_NAMES features
                  (+4: min, max, last, zero_fraction).  'catch24' + extra_stats
                  gives width 28.

    For full control over the feature set, call generate_node_features directly.
    """
    if mode not in _MODE_FEATURE_LISTS:
        raise ValueError(
            f"mode must be 'raw', 'stats', 'catch22' or 'catch24', got {mode!r}"
        )
    feature_names = list(_MODE_FEATURE_LISTS[mode])
    if extra_stats and mode in ('catch22', 'catch24'):
        feature_names += _EXTRA_STAT_NAMES
    return generate_node_features(window_values, feature_names)
