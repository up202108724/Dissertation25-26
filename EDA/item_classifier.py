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
      - erratic: ADI <= threshold AND CV > threshold  
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
            label = "erratic"
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
