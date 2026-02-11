"""
Classification of items into: Regular, Seasonal, Intermittent, New
Based on sales patterns and temporal characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import variation
from statsmodels.tsa.seasonal import STL


class ItemClassifier:
    """
    Classifier for retail items based on their sales patterns.
    
    Categories:
    - Regular: Consistent sales pattern, no strong seasonality
    - Seasonal: Periodic/cyclic sales pattern
    - Intermittent: Erratic/unpredictable sales pattern
    - New: Item with non-sales period at the beginning
    """
    
    def __init__(
        self,
        seasonal_strength_threshold: float = 0.5,
        cv_threshold: float = 0.8,
        initial_gap_fraction: float = 0.2,
        min_periods: int = 14,
    ):
        """
        Parameters
        ----------
        seasonal_strength_threshold : float
            Threshold for seasonality strength (0-1). Values above this indicate seasonal patterns.
        cv_threshold : float
            Coefficient of variation threshold to detect intermittence in cycle intervals.
        initial_gap_fraction : float
            If the fraction of zero-sales at the beginning exceeds this, mark as new.
        min_periods : int
            Minimum periods required for reliable seasonal detection.
        """
        self.seasonal_strength_threshold = seasonal_strength_threshold
        self.cv_threshold = cv_threshold
        self.initial_gap_fraction = initial_gap_fraction
        self.min_periods = min_periods
    
    def classify_item(
        self,
        series: pd.Series,
        item_id: Optional[int] = None,
    ) -> Dict:
        """
        Classify a single item's sales series.
        
        Parameters
        ----------
        series : pd.Series
            Sales values indexed by date/time, sorted chronologically.
        item_id : int, optional
            Item identifier for reference.
        
        Returns
        -------
        dict
            Contains 'label', 'item_id', and diagnostic information.
        """
        result = {
            'item_id': item_id,
            'label': None,
            'is_new': False,
            'is_seasonal': False,
            'is_intermittent': False,
            'seasonal_strength': None,
            'cycle_cv': None,
            'initial_gap_fraction': None,
            'n_observations': len(series),
        }
        
        # Basic validation
        if len(series) < self.min_periods:
            result['label'] = 'regular'  # insufficient data -> regular by default
            return result
        
        # 1. Check if NEW: significant non-sales period at the beginning
        result['initial_gap_fraction'] = self._detect_new_item(series)
        if result['initial_gap_fraction'] >= self.initial_gap_fraction:
            result['is_new'] = True
            result['label'] = 'new'
            return result  # Don't classify further
        
        # 2. Check if SEASONAL: periodicity in sales pattern
        result['seasonal_strength'] = self._detect_seasonality(series)
        if result['seasonal_strength'] is not None and \
           result['seasonal_strength'] >= self.seasonal_strength_threshold:
            result['is_seasonal'] = True
        
        # 3. Check if INTERMITTENT: erratic intervals between sales/non-sales cycles
        result['cycle_cv'] = self._detect_intermittence(series)
        if result['cycle_cv'] is not None and result['cycle_cv'] >= self.cv_threshold:
            result['is_intermittent'] = True
        
        # 4. Assign final label
        if result['is_seasonal']:
            result['label'] = 'seasonal'
        elif result['is_intermittent']:
            result['label'] = 'intermittent'
        else:
            result['label'] = 'regular'
        
        return result
    
    def _detect_new_item(self, series: pd.Series) -> float:
        """
        Detect if item is NEW: has significant non-sales at the beginning.
        
        Returns the fraction of zero-sales at the start before first non-zero sale.
        """
        # Find first non-zero sales
        nonzero_idx = (series > 0).idxmax()
        if not nonzero_idx:  # All zeros
            return 1.0
        
        first_sale_pos = series.index.get_loc(nonzero_idx)
        gap_fraction = first_sale_pos / len(series)
        return gap_fraction
    
    def _detect_seasonality(self, series: pd.Series) -> Optional[float]:
        """
        Detect seasonality using STL decomposition.
        
        Returns the seasonal strength (0-1):
        strength = 1 - Var(remainder) / Var(seasonal + remainder)
        """
        try:
            # Need sufficient data for STL
            if len(series) < 2 * self.min_periods:
                return None
            
            # Determine seasonal period (assume weekly for retail)
            period = min(7, len(series) // 4)
            if period < 2:
                return None
            
            # Fill NaN with 0 for STL
            filled_series = series.fillna(0)
            
            # STL decomposition
            stl = STL(filled_series, seasonal=period, robust=True)
            result = stl.fit()
            
            seasonal = result.seasonal.values
            residual = result.resid.values
            
            # Seasonal strength = 1 - Var(residual) / Var(seasonal + residual)
            var_residual = np.var(residual)
            var_seasonal_residual = np.var(seasonal + residual)
            
            if var_seasonal_residual == 0:
                return 0.0
            
            strength = 1.0 - (var_residual / var_seasonal_residual)
            strength = np.clip(strength, 0, 1)
            return strength
        except Exception:
            return None
    
    def _detect_intermittence(self, series: pd.Series) -> Optional[float]:
        """
        Detect intermittence by analyzing variability in cycle intervals.
        
        For both sales and non-sales cycles, compute interval lengths and 
        calculate coefficient of variation. High CV indicates intermittent pattern.
        """
        try:
            # Create binary indicator: 1 if sale, 0 if non-sale
            binary = (series > 0).astype(int).values
            
            if np.sum(binary) < 2:  # Too few sales
                return None
            
            # Find transitions: points where binary value changes
            transitions = np.where(np.diff(binary) != 0)[0] + 1
            
            if len(transitions) < 2:
                return None
            
            # Compute interval lengths between transitions
            interval_lengths = np.diff(transitions)
            
            if len(interval_lengths) < 2:
                return None
            
            # Coefficient of variation
            mean_interval = np.mean(interval_lengths)
            if mean_interval == 0:
                return 1.0  # Highly variable
            
            cv = np.std(interval_lengths) / mean_interval
            return cv
        except Exception:
            return None
    
    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all items in a dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: 'item_id', 'value' (sales), and a datetime index or date column.
        
        Returns
        -------
        pd.DataFrame
            Original df with added 'item_label' column.
        """
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        
        labels = []
        
        for item_id in df['item_id'].unique():
            item_series = df[df['item_id'] == item_id]['value'].sort_index()
            result = self.classify_item(item_series, item_id=item_id)
            labels.append(result)
        
        labels_df = pd.DataFrame(labels)
        
        # Merge back to original dataframe
        output_df = df.copy()
        output_df = output_df.merge(
            labels_df[['item_id', 'label']],
            on='item_id',
            how='left'
        )
        output_df = output_df.rename(columns={'label': 'item_label'})
        
        return output_df, labels_df
    
    def get_label_summary(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics of label distribution.
        """
        return labels_df[labels_df['label'].notna()].groupby('label').size().to_frame('count')


def classify_items(
    df: pd.DataFrame,
    seasonal_strength_threshold: float = 0.3,
    cv_threshold: float = 0.5,
    initial_gap_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to classify items in a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must have 'item_id', 'value' columns and datetime index.
    seasonal_strength_threshold : float
        Threshold for seasonal detection (0-1).
    cv_threshold : float
        Threshold for intermittent detection (CV of cycle intervals).
    initial_gap_fraction : float
        Threshold for new item detection.
    
    Returns
    -------
    df_classified : pd.DataFrame
        Original df with 'item_label' column added.
    summary : dict
        Summary statistics about label distribution.
    """
    classifier = ItemClassifier(
        seasonal_strength_threshold=seasonal_strength_threshold,
        cv_threshold=cv_threshold,
        initial_gap_fraction=initial_gap_fraction,
    )
    
    df_classified, labels_df = classifier.classify_dataframe(df)
    summary = {
        'label_counts': labels_df['label'].value_counts().to_dict(),
        'total_items': len(labels_df),
        'summary_stats': labels_df[['label', 'seasonal_strength', 'cycle_cv', 'initial_gap_fraction']].groupby('label').agg(['mean', 'min', 'max']),
    }
    
    # Drop pontual features
    df_classified = df_classified.drop(columns=['promotions', 'value'], errors='ignore')
    
    return df_classified, summary

def classify_items_adi_cv(
    df: pd.DataFrame,
    adi_threshold: float = 1.32,
    cv_threshold: float = 0.49,
    item_col: str = "item_id",
    value_col: str = "value",
    date_col: Optional[str] = None,          # if None, expects DatetimeIndex
    agg_per_date: Optional[str] = None,      # None | "sum" | "mean" (if duplicates per date)
    # NEW detection
    new_initial_gap_fraction: float = 0.2,   # >= this fraction of leading zeros => new
    # EOL detection
    eol_window: int = 56,                    # trailing window length (periods)
    eol_gap: int = 28,                       # last sale at least this many periods before end
    eol_max_nonzero_in_window: int = 1,      # activity allowed in the trailing window
    # labeling behavior
    override_label_for_new_eol: bool = False # if True, label='new'/'end_of_life' overrides quadrant
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    ADI/CV classification + flags for NEW and End-of-Life (EOL).

    Quadrants (Syntetos et al.):
      - smooth, seasonal, intermittent, lumpy

    Flags:
      - is_new: large initial zero gap (leading zeros)
      - is_end_of_life: long trailing gap + low activity in last window

    If override_label_for_new_eol=True, the final label becomes:
      'new' or 'end_of_life' (priority: end_of_life > new) else the quadrant label.
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

        # ---- NEW flag (leading zeros fraction before first non-zero)
        nz = np.flatnonzero(y > 0)
        if len(nz) == 0:
            first_sale_pos = None
            initial_gap_frac = 1.0
        else:
            first_sale_pos = int(nz[0])
            initial_gap_frac = first_sale_pos / n if n else 1.0
        is_new = initial_gap_frac >= new_initial_gap_fraction

        # ---- EOL flag (gap to end + low activity in last window)
        if len(nz) == 0:
            last_sale_pos = None
            gap_to_end = n
        else:
            last_sale_pos = int(nz[-1])
            gap_to_end = (n - 1) - last_sale_pos

        tail = y[-min(eol_window, n):] if n else np.array([])
        tail_nonzero = int((tail > 0).sum()) if n else 0
        is_eol = (gap_to_end >= eol_gap) and (tail_nonzero <= eol_max_nonzero_in_window)

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
            quad_label = "smooth"
        elif adi <= adi_threshold and cv > cv_threshold:
            quad_label = "seasonal"
        elif adi > adi_threshold and cv <= cv_threshold:
            quad_label = "intermittent"
        else:
            quad_label = "lumpy"

        # ---- Final label (optional override)
        if override_label_for_new_eol:
            if is_eol:
                label = "end_of_life"
            elif is_new:
                label = "new"
            else:
                label = quad_label
        else:
            label = quad_label

        rows.append({
            item_col: item_id,
            "label": label,
            "quad_label": quad_label,
            "adi": adi,
            "cv": cv,
            "non_zero_periods": non_zero_periods,
            "total_periods": total_periods,
            "is_new": bool(is_new),
            "initial_gap_fraction": float(initial_gap_frac),
            "is_end_of_life": bool(is_eol),
            "gap_to_end": int(gap_to_end),
            "tail_nonzero": int(tail_nonzero),
        })

    labels_df = pd.DataFrame(rows)

    

    # ✅ Preserve DatetimeIndex: map labels (no merge)
    output_df = df.copy()
    label_map = labels_df.set_index(item_col)["label"]
    output_df["item_label"] = output_df[item_col].map(label_map)

    # ✅ JSON-safe summary (no DataFrames inside the dict)
    summary_stats = (
        labels_df.groupby("label")[["adi", "cv", "initial_gap_fraction", "gap_to_end"]]
        .agg(["mean", "min", "max"])
    )
    # flatten MultiIndex columns -> strings
    summary_stats.columns = [f"{a}_{b}" for a, b in summary_stats.columns]
    summary_stats_json = summary_stats.reset_index().to_dict(orient="records")

    summary = {
        "label_counts": labels_df["label"].value_counts().to_dict(),
        "total_items": int(len(labels_df)),
        "summary_stats": summary_stats_json,   # ✅ JSON-friendly
        "flag_counts": {
            "is_new": int(labels_df["is_new"].sum()),
            "is_end_of_life": int(labels_df["is_end_of_life"].sum()),
        }
    }


    return output_df, labels_df, summary

