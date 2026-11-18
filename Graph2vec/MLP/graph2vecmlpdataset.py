# Kept for backwards compatibility: the windowing now lives in the baseline
# TDMLP dataset module (embeddings are an optional branch there).
from Baselines.MLP.tdmlp_dataset import WindowDataset, make_windows

__all__ = ["WindowDataset", "make_windows"]
