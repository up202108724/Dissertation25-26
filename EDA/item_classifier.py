"""
Classification of items into: Regular, Seasonal, Intermittent, New
Based on sales patterns and temporal characteristics.
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import variation
from statsmodels.tsa.seasonal import STL


class ItemClassifier:
    
    def __init__(self) -> None:
        self.w1, self.w2, self.w3 = 0.1, 1.0, 0.5
        self.max_dist = 0.5

    
    def set_weights(self, w1:float, w2:float, w3:float, max_dist:float) -> None:
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.max_dist = max_dist
        
        
    def is_irrelevant(self, ts: DataFrame, scat_ABC: tuple, target_var: str) -> bool:
        _, B, C = scat_ABC
        cond1 = ts[target_var].sum() < self.w1 * B
        cond2 = ts[target_var].max() < self.w2 * C
        return cond1 & cond2
    
    
    def is_small(self, ts: DataFrame, scat_ABC, target_var: str) -> bool:
        _, B, _ = scat_ABC
        return (self.w1 * B) <= ts[target_var].sum() <= (self.w3 * B)

    
    def is_representative(self, ts: DataFrame, scat_ABC: tuple, scat_stats: tuple, target_var: str) -> bool:
        
        def close_factors(f1, f2, f3):
            res = max(f1, f2, f3) / min(f1, f2, f3)
            return res <= 1 + self.max_dist # 1.5 # 1.3
        
        _, B, _ = scat_ABC
        scat_mean, scat_std, scat_amp = scat_stats
        mean_factor = ts[target_var].mean() / scat_mean
        std_factor = ts[target_var].std() / scat_std
        amp_factor = (ts[target_var].max() - ts[target_var].min()) / scat_amp
        return (ts[target_var].sum() > self.w3 * B) & close_factors(mean_factor, std_factor, amp_factor) # & close_factors(mean_factor, amp_factor)
    
    
    def select_classification(self, ts: DataFrame, scat_ABC: tuple, scat_stats: tuple, target_var: str) -> str:
        if self.is_representative(ts, scat_ABC, scat_stats, target_var): return "representative"
        if self.is_irrelevant(ts, scat_ABC, target_var): return "irrelevant"
        if self.is_small(ts, scat_ABC, target_var): return "small"
        return "independent"
    

 

def classify_items_adi_cv(
    df: pd.DataFrame,
    adi_threshold: float = 1.32,
    cv_threshold: float = 0.49,
    item_col: str = "item_id",
    value_col: str = "value",
    date_col: Optional[str] = None,          
    agg_per_date: Optional[str] = None,      # None | "sum" | "mean" (if duplicates per date)
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    ADI/CV classification (Syntetos et al.).

    Quadrants:
      - smooth: ADI <= threshold AND CV <= threshold
      - seasonal: ADI <= threshold AND CV > threshold  
      - intermittent: ADI > threshold AND CV <= threshold
      - lumpy: ADI > threshold AND CV > threshold
    """

    work = df.copy()

    # --- Ensure we can access dates
    if date_col is not None:
        work[date_col] = pd.to_datetime(work[date_col])
        work = work.sort_values([item_col, date_col])
    else:
        if not isinstance(work.index, pd.DatetimeIndex):
            raise ValueError("Provide date_col or ensure df has a DatetimeIndex.")
        work = work.sort_index()

    rows = []

    for item_id in work[item_col].dropna().unique():
        item_df = work.loc[work[item_col] == item_id, :]

        # Build series indexed by date
        if date_col is not None:
            s = item_df.set_index(date_col)[value_col].sort_index()
        else:
            s = item_df[value_col].sort_index()

        # Optional aggregation per date (if duplicates)
        if agg_per_date in ("sum", "mean"):
            if agg_per_date == "sum":
                s = s.groupby(level=0).sum()
            else:
                s = s.groupby(level=0).mean()

        y = s.to_numpy()
        n = len(y)

        # ---- ADI
        non_zero_periods = int((y > 0).sum())
        total_periods = int(n)

        adi = np.inf if non_zero_periods == 0 else (total_periods / non_zero_periods)

        # ---- CV on non-zero demands
        nonzero_demands = y[y > 0]
        if len(nonzero_demands) > 1:
            cv = float(np.std(nonzero_demands, ddof=1) / np.mean(nonzero_demands))
        elif len(nonzero_demands) == 1:
            cv = 0.0
        else:
            cv = np.inf

        # ---- Quadrant label
        if adi <= adi_threshold and cv <= cv_threshold:
            label = "smooth"
        elif adi <= adi_threshold and cv > cv_threshold:
            label = "seasonal"
        elif adi > adi_threshold and cv <= cv_threshold:
            label = "intermittent"
        else:
            label = "lumpy"

        rows.append({
            item_col: item_id,
            "label": label,
            "adi": adi,
            "cv": cv,
            "non_zero_periods": non_zero_periods,
            "total_periods": total_periods,
        })

    labels_df = pd.DataFrame(rows)

    # ✅ Preserve DatetimeIndex: map labels
    output_df = df.copy()
    label_map = labels_df.set_index(item_col)["label"]
    output_df["item_label"] = output_df[item_col].map(label_map)

    # ✅ JSON-safe summary
    summary_stats = (
        labels_df.groupby("label")[["adi", "cv"]]
        .agg(["mean", "min", "max"])
    )
    # flatten MultiIndex columns -> strings
    summary_stats.columns = [f"{a}_{b}" for a, b in summary_stats.columns]
    summary_stats_json = summary_stats.reset_index().to_dict(orient="records")

    summary = {
        "label_counts": labels_df["label"].value_counts().to_dict(),
        "total_items": int(len(labels_df)),
        "summary_stats": summary_stats_json,
    }

    return output_df, labels_df, summary


def classify_items_item_classifier(
    df: pd.DataFrame,
    classifier: Optional[ItemClassifier] = None,
    item_col: str = "item_id",
    value_col: str = "value",
    date_col: Optional[str] = None,          
    agg_per_date: Optional[str] = None,     
    output_label_col: str = "item_label",
    a_quantile: float = 0.80,
    b_quantile: float = 0.50,
    c_quantile: float = 0.50,
    stats_agg: str = "median",            
    epsilon: float = 1e-8,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Label items using the rule-based `ItemClassifier`.

    This mirrors `classify_items_adi_cv`'s interface/outputs:
    - returns `output_df` (same shape as input) with `output_label_col` mapped per item
    - returns `labels_df` with one row per item and the computed label + diagnostics
    - returns a JSON-safe `summary` dict

    Notes on calibration
    --------------------
    `ItemClassifier` expects two calibration tuples:
      - `scat_ABC = (A, B, C)` where B is a "typical" total-sales scale and C is a "typical" max-sales scale.
      - `scat_stats = (mean, std, amp)` computed across items.

    Since your project currently doesn't define how these are computed externally, this function derives them
    from per-item aggregates via quantiles / central tendency. If you later decide on a different definition,
    you can keep the labeling loop and swap out the calibration block.
    """

    if classifier is None:
        classifier = ItemClassifier()

    work = df.copy()

    # --- Ensure we can access dates and keep a consistent order
    if date_col is not None:
        work[date_col] = pd.to_datetime(work[date_col])
        work = work.sort_values([item_col, date_col])
    else:
        if not isinstance(work.index, pd.DatetimeIndex):
            raise ValueError("Provide date_col or ensure df has a DatetimeIndex.")
        work = work.sort_index()

    if item_col not in work.columns:
        raise ValueError(f"Missing required column: {item_col}")
    if value_col not in work.columns:
        raise ValueError(f"Missing required column: {value_col}")

    # Ensure numeric values
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)

    # --- Per-item aggregates for calibration and diagnostics
    per_item = (
        work.groupby(item_col)[value_col]
        .agg(
            n_obs="count",
            total_sales="sum",
            mean_sales="mean",
            std_sales="std",
            max_sales="max",
            min_sales="min",
        )
        .reset_index()
    )
    per_item["amp_sales"] = per_item["max_sales"] - per_item["min_sales"]

    # Handle edge cases: std can be NaN for single-observation items
    per_item["std_sales"] = per_item["std_sales"].fillna(0.0)

    if len(per_item) == 0:
        labels_df = pd.DataFrame(columns=[item_col, "label"])
        output_df = df.copy()
        output_df[output_label_col] = np.nan
        summary = {"label_counts": {}, "total_items": 0, "summary_stats": []}
        return output_df, labels_df, summary

    # --- Build scat_ABC and scat_stats
    def _q(series: pd.Series, q: float) -> float:
        q = float(q)
        q = 0.0 if q < 0.0 else 1.0 if q > 1.0 else q
        return float(series.quantile(q))

    A = _q(per_item["total_sales"], a_quantile)
    B = _q(per_item["total_sales"], b_quantile)
    C = _q(per_item["max_sales"], c_quantile)
    scat_ABC = (A, B, C)

    if stats_agg not in ("median", "mean"):
        raise ValueError("stats_agg must be 'median' or 'mean'")

    if stats_agg == "median":
        scat_mean = float(per_item["mean_sales"].median())
        scat_std = float(per_item["std_sales"].median())
        scat_amp = float(per_item["amp_sales"].median())
    else:
        scat_mean = float(per_item["mean_sales"].mean())
        scat_std = float(per_item["std_sales"].mean())
        scat_amp = float(per_item["amp_sales"].mean())

    # Avoid division by zero in `is_representative`
    scat_stats = (
        max(abs(scat_mean), epsilon),
        max(abs(scat_std), epsilon),
        max(abs(scat_amp), epsilon),
    )

    # --- Label each item
    rows: List[Dict] = []
    for current_item_id in work[item_col].dropna().unique():
        item_df = work.loc[work[item_col] == current_item_id, :]

        # Build per-item time series as a DataFrame; optionally aggregate duplicates per date
        if date_col is not None:
            ts = item_df[[date_col, value_col]].sort_values(date_col)
            if agg_per_date in ("sum", "mean"):
                ts = (
                    ts.groupby(date_col, as_index=False)[value_col]
                    .sum() if agg_per_date == "sum" else ts.groupby(date_col, as_index=False)[value_col].mean()
                )
            ts = ts.rename(columns={date_col: "date"})
        else:
            # DatetimeIndex
            ts = item_df[[value_col]].copy()
            if agg_per_date in ("sum", "mean"):
                ts[value_col] = (
                    ts[value_col].groupby(level=0).sum()
                    if agg_per_date == "sum"
                    else ts[value_col].groupby(level=0).mean()
                )

        label = classifier.select_classification(ts, scat_ABC=scat_ABC, scat_stats=scat_stats, target_var=value_col)

        # Attach diagnostics from precomputed aggregates
        agg_row = per_item.loc[per_item[item_col] == current_item_id].iloc[0]
        rows.append({
            item_col: current_item_id,
            "label": label,
            "n_obs": int(agg_row["n_obs"]),
            "total_sales": float(agg_row["total_sales"]),
            "mean_sales": float(agg_row["mean_sales"]),
            "std_sales": float(agg_row["std_sales"]),
            "max_sales": float(agg_row["max_sales"]),
            "amp_sales": float(agg_row["amp_sales"]),
        })

    labels_df = pd.DataFrame(rows)

    # Map back onto the original df (preserve original index)
    output_df = df.copy()
    label_map = labels_df.set_index(item_col)["label"]
    output_df[output_label_col] = output_df[item_col].map(label_map)

    # JSON-safe summary
    label_counts = labels_df["label"].value_counts().to_dict()
    summary_stats = (
        labels_df
        .groupby("label")[["n_obs", "total_sales", "mean_sales", "std_sales", "max_sales", "amp_sales"]]
        .agg(["mean", "min", "max"])
    )
    summary_stats.columns = [f"{a}_{b}" for a, b in summary_stats.columns]
    summary_stats_json = summary_stats.reset_index().to_dict(orient="records")

    summary = {
        "label_counts": label_counts,
        "total_items": int(len(labels_df)),
        "summary_stats": summary_stats_json,
        "calibration": {
            "scat_ABC": {"A_total_sales": float(A), "B_total_sales": float(B), "C_max_sales": float(C)},
            "scat_stats": {"mean": float(scat_stats[0]), "std": float(scat_stats[1]), "amp": float(scat_stats[2])},
            "params": {
                "a_quantile": float(a_quantile),
                "b_quantile": float(b_quantile),
                "c_quantile": float(c_quantile),
                "stats_agg": stats_agg,
                "agg_per_date": agg_per_date,
            },
        },
    }

    return output_df, labels_df, summary

