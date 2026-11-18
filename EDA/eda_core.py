"""
Shared EDA logic: loading, calendar-grid imputation, coverage, ADI/CV
classification, sales summaries, top-product selection, store closures,
holidays and promotions.

Every function works on the long panel (one row per item, store, date).
"""

import os

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths / column names
# -----------------------------------------------------------------------------
EDA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EDA_DIR)
DATASET_DIR = os.path.join(REPO_DIR, "dataset")
RAW_PATH = os.path.join(DATASET_DIR, "data_andre.feather")
FULFILLED_PATH = os.path.join(DATASET_DIR, "data_andre_fulfilled.feather")
CLASSIFIED_PATH = os.path.join(DATASET_DIR, "data_andre_classified.feather")

ITEM_COL, STORE_COL, DATE_COL, VALUE_COL = "item_id", "store_id", "date", "value"
KEYS = [ITEM_COL, STORE_COL]
LABEL_COL = "adi_cv_label"
CATEGORY_COLS = ("cat_label", "sdep_label", "dep_label", "dmn_label")
PROMO_PREFIXES = ("promo_type_", "promo_value_")

# Syntetos et al. (2005) cut-offs.
ADI_THRESHOLD = 1.32
CV_THRESHOLD = 0.49
LABEL_ORDER = ["smooth", "erratic", "intermittent", "lumpy"]


def top_products_path(min_total_sales):
    return os.path.join(DATASET_DIR, f"top_{min_total_sales}.feather")


# -----------------------------------------------------------------------------
# Loading / imputation
# -----------------------------------------------------------------------------
def load_panel(path=RAW_PATH):
    """Long panel with ``date`` as a normalised datetime column, sorted by pair and date."""
    df = pd.read_feather(path)
    if DATE_COL not in df.columns and df.index.name == DATE_COL:
        df = df.reset_index()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.normalize()
    return df.sort_values(KEYS + [DATE_COL]).reset_index(drop=True)


def promo_columns(df, kind=None):
    """Promotion columns; ``kind`` = 'type' | 'value' | None (both)."""
    prefixes = PROMO_PREFIXES if kind is None else (f"promo_{kind}_",)
    return [c for c in df.columns if c.startswith(prefixes)]


def category_columns(df):
    return [c for c in df.columns
            if c in CATEGORY_COLS or c.startswith(tuple(f"{p}_" for p in CATEGORY_COLS))]


def coverage_report(df):
    """Per (item, store): first/last date, observed days and gaps against the global range."""
    gmin, gmax = df[DATE_COL].min(), df[DATE_COL].max()
    expected = (gmax - gmin).days + 1
    cov = (df.groupby(KEYS)[DATE_COL]
             .agg(date_min="min", date_max="max", n_days="nunique")
             .reset_index())
    cov["missing_days"] = expected - cov["n_days"]
    cov["missingness_rate"] = (cov["missing_days"] / expected * 100).round(2)
    cov["days_missing_before"] = (cov["date_min"] - gmin).dt.days
    cov["days_missing_after"] = (gmax - cov["date_max"]).dt.days
    return cov


def fill_calendar_grid(df):
    """Complete daily grid for every (item, store) over the global date range.

    Missing days get ``value = 0`` and promotions = 0; category labels are
    carried forward/backward within each pair.
    """
    dates = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    grid = df[KEYS].drop_duplicates().merge(pd.DataFrame({DATE_COL: dates}), how="cross")
    out = grid.merge(df, on=KEYS + [DATE_COL], how="left")

    out[VALUE_COL] = out[VALUE_COL].fillna(0)
    promos = promo_columns(out)
    out[promos] = out[promos].fillna(0)
    cats = category_columns(out)
    if cats:
        out[cats] = out.groupby(KEYS)[cats].transform(lambda s: s.ffill().bfill())
    return out


# -----------------------------------------------------------------------------
# ADI / CV classification
# -----------------------------------------------------------------------------
def adi_cv(values):
    """ADI = periods / non-zero periods; CV = std/mean of the non-zero demands."""
    y = np.asarray(values, dtype=float)
    nz = y[y > 0]
    adi = np.inf if len(nz) == 0 else len(y) / len(nz)
    if len(nz) > 1:
        cv = float(np.std(nz, ddof=1) / np.mean(nz))
    else:
        cv = 0.0 if len(nz) == 1 else np.inf
    return adi, cv


def adi_cv_label(adi, cv, adi_threshold=ADI_THRESHOLD, cv_threshold=CV_THRESHOLD):
    if adi <= adi_threshold:
        return "smooth" if cv <= cv_threshold else "erratic"
    return "intermittent" if cv <= cv_threshold else "lumpy"


def classify_adi_cv(df, adi_threshold=ADI_THRESHOLD, cv_threshold=CV_THRESHOLD):
    """One row per (item, store): label, adi, cv, non_zero_periods, total_periods."""
    rows = []
    for (item_id, store_id), g in df.sort_values(DATE_COL).groupby(KEYS, sort=True):
        y = g[VALUE_COL].to_numpy()
        adi, cv = adi_cv(y)
        rows.append({
            ITEM_COL: item_id, STORE_COL: store_id,
            LABEL_COL: adi_cv_label(adi, cv, adi_threshold, cv_threshold),
            "adi": adi, "cv": cv,
            "non_zero_periods": int((y > 0).sum()), "total_periods": len(y),
        })
    return pd.DataFrame(rows)


def add_labels(df, labels):
    """Broadcast each pair's ADI/CV label onto all its rows."""
    return df.merge(labels[KEYS + [LABEL_COL]], on=KEYS, how="left")


# -----------------------------------------------------------------------------
# Sales summaries / product selection
# -----------------------------------------------------------------------------
def sales_summary(df, labels=None):
    """Per (item, store): total / mean / median sales, non-zero days and zero rate."""
    s = (df.assign(_nz=df[VALUE_COL] > 0)
           .groupby(KEYS)
           .agg(total_sales=(VALUE_COL, "sum"), mean_sales=(VALUE_COL, "mean"),
                median_sales=(VALUE_COL, "median"), n_days=(VALUE_COL, "size"),
                nonzero_days=("_nz", "sum"))
           .reset_index())
    s["zero_rate"] = 1 - s["nonzero_days"] / s["n_days"]
    if labels is not None:
        s = s.merge(labels[KEYS + [LABEL_COL, "adi", "cv"]], on=KEYS, how="left")
    return s


def select_top_products(summary, labels=("smooth", "erratic"), min_total_sales=None,
                        min_mean_sales=None, max_zero_rate=None, sort_by="total_sales"):
    """Filter a ``sales_summary`` (with labels) and sort it descending by ``sort_by``."""
    mask = pd.Series(True, index=summary.index)
    if labels is not None:
        mask &= summary[LABEL_COL].isin(labels)
    if min_total_sales is not None:
        mask &= summary["total_sales"] >= min_total_sales
    if min_mean_sales is not None:
        mask &= summary["mean_sales"] >= min_mean_sales
    if max_zero_rate is not None:
        mask &= summary["zero_rate"] <= max_zero_rate
    return summary[mask].sort_values(sort_by, ascending=False).reset_index(drop=True)


def rows_for_pairs(df, pairs):
    """Rows of ``df`` whose (item, store) appears in ``pairs``."""
    return df.merge(pairs[KEYS].drop_duplicates(), on=KEYS, how="inner")


def product_rows(df, item_id, store_id=None):
    mask = df[ITEM_COL] == item_id
    if store_id is not None:
        mask &= df[STORE_COL] == store_id
    return df[mask].sort_values(DATE_COL).reset_index(drop=True)


def skus_per(df, col):
    """Number of distinct items per value of ``col`` (e.g. 'dep_label', 'cat_label')."""
    return df.groupby(col)[ITEM_COL].nunique().sort_values(ascending=False)


# -----------------------------------------------------------------------------
# Stores, holidays, promotions
# -----------------------------------------------------------------------------
def daily_store_sales(df):
    return df.groupby([STORE_COL, DATE_COL], as_index=False)[VALUE_COL].sum()


def store_closures(df, store_id):
    """Days in the global range with no sales in the store, grouped into consecutive periods."""
    daily = daily_store_sales(df)
    daily = daily[daily[STORE_COL] == store_id]
    all_dates = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    open_days = pd.DatetimeIndex(daily.loc[daily[VALUE_COL] > 0, DATE_COL])
    closed = pd.Series(all_dates.difference(open_days))

    if closed.empty:
        periods = pd.DataFrame(columns=["start", "end", "days"])
    else:
        run = (closed.diff() != pd.Timedelta(days=1)).cumsum()
        periods = (closed.groupby(run).agg(start="min", end="max", days="size")
                         .sort_values("days", ascending=False).reset_index(drop=True))
    return {
        "store_id": store_id,
        "total_days": len(all_dates),
        "open_days": len(open_days),
        "closed_days": len(closed),
        "closure_rate": len(closed) / len(all_dates),
        "periods": periods,
    }


def us_holidays_in_range(df):
    import holidays
    dmin, dmax = df[DATE_COL].min(), df[DATE_COL].max()
    cal = holidays.UnitedStates(years=range(dmin.year, dmax.year + 1), observed=True)
    return pd.Series({pd.Timestamp(d): name for d, name in cal.items()
                      if dmin <= pd.Timestamp(d) <= dmax}).sort_index()


def holiday_store_sales(df):
    """Daily store sales on US (observed) holidays, with the holiday name."""
    hol = us_holidays_in_range(df)
    daily = daily_store_sales(df)
    daily = daily[daily[DATE_COL].isin(hol.index)].copy()
    daily["holiday"] = daily[DATE_COL].map(hol)
    return daily.reset_index(drop=True)


def promo_summary(df_product):
    """Sales with vs without an active promotion, and total sales per active promo type."""
    active = [c for c in promo_columns(df_product, "type") if df_product[c].sum() > 0]
    if not active:
        return active, None, None
    any_promo = df_product[active].max(axis=1).gt(0).rename("any_promo_active")
    by_flag = (df_product.groupby(any_promo)[VALUE_COL]
                         .agg(["mean", "median", "min", "max", "count", "sum"]).round(2))
    by_type = pd.Series({c.replace("promo_type_", ""): df_product.loc[df_product[c] > 0, VALUE_COL].sum()
                         for c in active}, name="total_sales")
    return active, by_flag, by_type
