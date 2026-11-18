import numpy as np
import torch
from typing import Tuple
import pandas as pd
import holidays
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.nn as nn
import networkx as nx
from tslearn.metrics import cdist_dtw
    
def _proximity_decay(
    dates: pd.Series,
    month: int,
    day: int,
    tau_before: float = 10.0,
    tau_after: float = 3.0,
):
    """Asymmetric exp-decay proximity to an annual fixed-date event.

    Returns a value in [0, 1]:
      - 1.0  on the event day itself.
      - exp(-days_to  / tau_before) in the days leading up to the event.
      - exp(-days_from / tau_after)  in the days after the event.

    Asymmetry reflects retail demand patterns: demand builds gradually before
    a holiday (large tau_before) and drops sharply after (small tau_after).
    Defaults: tau_before=10 (~2-week ramp), tau_after=3 (~3-day tail).
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    years = dates.dt.year
    this_year = pd.to_datetime(dict(year=years, month=month, day=day))
    next_year = pd.to_datetime(dict(year=years + 1, month=month, day=day))
    prev_year = pd.to_datetime(dict(year=years - 1, month=month, day=day))

    # Signed distance to nearest occurrence: negative = before event, positive = after.
    d_this = (dates - this_year).dt.days.values   # positive after this year's event
    d_next = (dates - next_year).dt.days.values   # always negative (before)
    d_prev = (dates - prev_year).dt.days.values   # always positive (after)

    # Pick the occurrence that minimises |distance|.
    candidates = np.stack([d_this, d_next, d_prev], axis=1)
    idx = np.argmin(np.abs(candidates), axis=1)
    signed = candidates[np.arange(len(candidates)), idx]   # <0 before, >0 after, 0 on day

    tau = np.where(signed <= 0, tau_before, tau_after)
    return np.exp(-np.abs(signed) / tau).astype(np.float32)


def _proximity_decay_dynamic(
    dates: pd.Series,
    holiday_map,
    holiday_name: str,
    tau_before: float = 7.0,
    tau_after: float = 3.0,
    offset_days: int = 0,
):
    """Asymmetric exp-decay proximity for a moving holiday (e.g. Thanksgiving).

    Looks up every occurrence of *holiday_name* in *holiday_map*, optionally
    shifts by *offset_days* (positive = later, used for Black Friday = +1),
    then applies the same asymmetric decay as `_proximity_decay`.

    Defaults: tau_before=7 (~1-week ramp), tau_after=3 (~3-day tail).
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    event_dates = pd.to_datetime(sorted(
        d + pd.Timedelta(days=offset_days)
        for d, name in holiday_map.items()
        if name == holiday_name
    ))
    if len(event_dates) == 0:
        return np.zeros(len(dates), dtype=np.float32)

    dates_arr = dates.values.astype("datetime64[D]")
    events_arr = event_dates.values.astype("datetime64[D]")

    # For each date find the nearest event occurrence.
    signed_days = np.array([
        min(
            ((d - e) / np.timedelta64(1, "D") for e in events_arr),
            key=abs,
        )
        for d in dates_arr
    ], dtype=float)

    tau = np.where(signed_days <= 0, tau_before, tau_after)
    return np.exp(-np.abs(signed_days) / tau).astype(np.float32)
def _signed_days_to_event(dates: pd.Series, month: int, day: int, clip: int = 30):
    """Return (days_to_next, days_from_prev) for an annual fixed-date event.

    Both are non-negative integers clipped to [0, clip]. `days_to_next` is 0 on
    the event itself (and `days_from_prev` is also 0 that day).
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    years = dates.dt.year
    this_year = pd.to_datetime(dict(year=years, month=month, day=day))
    next_year = pd.to_datetime(dict(year=years + 1, month=month, day=day))
    prev_year = pd.to_datetime(dict(year=years - 1, month=month, day=day))

    days_to = np.where(
        this_year.values >= dates.values,
        (this_year - dates).dt.days.values,
        (next_year - dates).dt.days.values,
    )
    days_from = np.where(
        this_year.values <= dates.values,
        (dates - this_year).dt.days.values,
        (dates - prev_year).dt.days.values,
    )
    return np.clip(days_to, 0, clip).astype(int), np.clip(days_from, 0, clip).astype(int)
