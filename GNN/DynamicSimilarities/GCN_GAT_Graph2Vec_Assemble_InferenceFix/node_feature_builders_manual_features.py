from __future__ import annotations

import numpy as np
from _catch24_manual import catch22_all


_CATCH22_WIDTH = 22
_CATCH24_WIDTH = 24          # 22 catch22 + DN_Mean + DN_Spread_Std


# ── catch block helper ─────────────────────────────────────────────────────
def _compute_catch_block(window_values: np.ndarray, catch24: bool) -> np.ndarray:
    """(n_nodes, 22|24) catch22/24 features; degenerate series → 0."""
    n_nodes = window_values.shape[0]
    width = _CATCH24_WIDTH if catch24 else _CATCH22_WIDTH
    feats = np.zeros((n_nodes, width), dtype=np.float32)
    for i in range(n_nodes):
        ts = np.asarray(window_values[i], dtype=np.float64)
        try:
            vals = catch22_all(ts, catch24=catch24)["values"]
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
# Each preset is an explicit, ordered list of feature keys.  To add a new
# combination, register it here — every call site resolves through
# ``resolve_feature_names`` so build, inference-patch and ablation widths stay
# in sync automatically.  Presets ending in an explicit stat list (e.g.
# 'catch24_minmax') are self-contained: ``extra_stats`` does NOT append to them.
_MODE_FEATURE_LISTS: dict[str, list[str]] = {
    'raw':            ['raw'],
    'stats':          ['mean', 'std', 'min', 'max', 'first', 'last', 'slope', 'sum'],
    'catch22':        ['catch22'],
    'catch24':            ['catch24'],
    'catch24_minmax':     ['catch24', 'min', 'max'],
    'catch24_minmaxlast': ['catch24', 'min', 'max', 'last'],
}

# Bare catch presets get ``_EXTRA_STAT_NAMES`` auto-appended when
# extra_stats=True.  Explicit combos (anything else) are taken verbatim.
_EXTRA_STATS_MODES: set[str] = {'catch22', 'catch24'}

# Extra scale/level features appended to the bare catch presets when
# extra_stats=True.  Edit this list to add/remove extra stats — widths stay in
# sync automatically.
_EXTRA_STAT_NAMES: list[str] = ['min', 'max', 'zero_fraction']

# Kept for callers that import the constant directly (e.g. gcn_gat_assmble.py).
_EXTRA_STATS_WIDTH: int = sum(_FEATURE_WIDTHS[n] for n in _EXTRA_STAT_NAMES)


def resolve_feature_names(mode: str, extra_stats: bool = True) -> list[str]:
    """
    Single source of truth mapping a mode preset → ordered feature-name list.

    Used by ``window_node_features`` (dataset build + inference patch) and by
    the runner's width calculation (ablation dummy graphs), so every site that
    depends on the node-feature layout agrees by construction.

    mode        : any key of ``_MODE_FEATURE_LISTS``.
    extra_stats : appends ``_EXTRA_STAT_NAMES`` only for the bare catch presets
                  ('catch22', 'catch24'); ignored for explicit combos.
    """
    if mode not in _MODE_FEATURE_LISTS:
        raise ValueError(
            f"Unknown node-feature mode {mode!r}. "
            f"Available: {sorted(_MODE_FEATURE_LISTS)}"
        )
    names = list(_MODE_FEATURE_LISTS[mode])
    if extra_stats and mode in _EXTRA_STATS_MODES:
        names += _EXTRA_STAT_NAMES
    return names


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

    mode        : any key of ``_MODE_FEATURE_LISTS`` ('raw' | 'stats' |
                  'catch22' | 'catch24' | 'catch24_minmax' | ...).
    extra_stats : bare catch presets only — appends _EXTRA_STAT_NAMES
                  (min, max, zero_fraction).  'catch24' + extra_stats gives
                  width 27.  Explicit combos like 'catch24_minmax' are taken
                  verbatim regardless of this flag.

    For full control over the feature set, call generate_node_features directly.
    """
    feature_names = resolve_feature_names(mode, extra_stats=extra_stats)
    return generate_node_features(window_values, feature_names)
