"""
Classifies every (item_id, store_id) pair in data_andre_fulfilled.feather
using the ADI/CV scheme (Syntetos et al.):

  smooth       — ADI <= 1.32 AND CV <= 0.49
  erratic      — ADI <= 1.32 AND CV >  0.49
  intermittent — ADI >  1.32 AND CV <= 0.49
  lumpy        — ADI >  1.32 AND CV >  0.49

Output: data_andre_classified.feather
  All original columns + `adi_cv_label` (per pair, broadcast to every row).
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "..", "dataset")
INPUT_PATH   = os.path.join(DATA_DIR, "data_andre_fulfilled.feather")
OUTPUT_PATH  = os.path.join(DATA_DIR, "data_andre_classified.feather")

DATE_COL     = "date"
VALUE_COL    = "value"
ITEM_COL     = "item_id"
STORE_COL    = "store_id"
LABEL_COL    = "adi_cv_label"

ADI_THRESHOLD = 1.32
CV_THRESHOLD  = 0.49

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_feather(INPUT_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values([ITEM_COL, STORE_COL, DATE_COL]).reset_index(drop=True)

print(f"Loaded {len(df):,} rows, {df[[ITEM_COL, STORE_COL]].drop_duplicates().shape[0]:,} (item, store) pairs.")

# ---------------------------------------------------------------------------
# Classify each (item_id, store_id) pair
# ---------------------------------------------------------------------------
rows = []

for (item_id, store_id), group in df.groupby([ITEM_COL, STORE_COL], sort=False):
    y = group[VALUE_COL].to_numpy()

    # ADI: total periods / number of non-zero periods
    non_zero = int((y > 0).sum())
    adi = np.inf if non_zero == 0 else len(y) / non_zero

    # CV: std / mean of non-zero demands only
    nz_vals = y[y > 0]
    if len(nz_vals) > 1:
        cv = float(np.std(nz_vals, ddof=1) / np.mean(nz_vals))
    elif len(nz_vals) == 1:
        cv = 0.0
    else:
        cv = np.inf

    if adi <= ADI_THRESHOLD and cv <= CV_THRESHOLD:
        label = "smooth"
    elif adi <= ADI_THRESHOLD and cv > CV_THRESHOLD:
        label = "erratic"
    elif adi > ADI_THRESHOLD and cv <= CV_THRESHOLD:
        label = "intermittent"
    else:
        label = "lumpy"

    rows.append({
        ITEM_COL:  item_id,
        STORE_COL: store_id,
        LABEL_COL: label,
        "adi":     round(adi, 4),
        "cv":      round(cv, 4),
    })

labels_df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Merge label back onto every row
# ---------------------------------------------------------------------------
df = df.merge(labels_df[[ITEM_COL, STORE_COL, LABEL_COL]], on=[ITEM_COL, STORE_COL], how="left")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nLabel distribution (unique pairs):")
print(labels_df[LABEL_COL].value_counts().to_string())
print(f"\nADI/CV stats per label:")
print(labels_df.groupby(LABEL_COL)[["adi", "cv"]].describe().round(3).to_string())

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df.to_feather(OUTPUT_PATH)
print(f"\nSaved → {OUTPUT_PATH}  ({len(df):,} rows, {df.shape[1]} columns)")
print(f"Columns: {list(df.columns)}")
