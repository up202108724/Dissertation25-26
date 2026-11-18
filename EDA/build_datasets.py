"""
Builds the datasets used by the models, from the raw panel:

  1. fill      data_andre.feather           -> data_andre_fulfilled.feather
               complete daily grid per (item, store): missing days -> value 0,
               promotions 0, category labels carried forward/backward.
  2. classify  data_andre_fulfilled.feather -> data_andre_classified.feather
               adds `adi_cv_label` (smooth / erratic / intermittent / lumpy,
               Syntetos et al.: ADI 1.32, CV 0.49) to every row.
  3. top       data_andre_classified.feather -> top_<MIN_TOTAL_SALES>.feather
               rows of the smooth + erratic pairs with total sales >= MIN_TOTAL_SALES.

Usage (from anywhere):
  python EDA/build_datasets.py                     # all steps
  python EDA/build_datasets.py --steps classify top
  python EDA/build_datasets.py --out-dir some/dir  # write elsewhere (e.g. to compare)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eda_core as core  # noqa: E402

TOP_LABELS = ("smooth", "erratic")
MIN_TOTAL_SALES = 12500
STEPS = ("fill", "classify", "top")


def step_fill(out_dir):
    raw = core.load_panel(core.RAW_PATH)
    cov = core.coverage_report(raw)
    print(f"Raw: {len(raw):,} rows, {len(cov):,} (item, store) pairs, "
          f"{raw[core.DATE_COL].min().date()} -> {raw[core.DATE_COL].max().date()}")
    print(f"  incomplete pairs: {(cov['missing_days'] > 0).sum():,} "
          f"(started late: {(cov['days_missing_before'] > 0).sum():,}, "
          f"ended early: {(cov['days_missing_after'] > 0).sum():,})")

    filled = core.fill_calendar_grid(raw)
    nans = filled.isna().sum()
    nans = nans[nans > 0]
    print(f"Filled: {len(filled):,} rows (+{len(filled) - len(raw):,} imputed zero days)")
    print(f"  remaining NaNs: {nans.to_dict() if len(nans) else 'none'}")
    return _save(filled, out_dir, core.FULFILLED_PATH)


def step_classify(out_dir):
    df = core.load_panel(os.path.join(out_dir, os.path.basename(core.FULFILLED_PATH)))
    labels = core.classify_adi_cv(df)
    print("ADI/CV labels (pairs):")
    print(labels[core.LABEL_COL].value_counts().reindex(core.LABEL_ORDER, fill_value=0).to_string())
    return _save(core.add_labels(df, labels), out_dir, core.CLASSIFIED_PATH)


def step_top(out_dir):
    df = core.load_panel(os.path.join(out_dir, os.path.basename(core.CLASSIFIED_PATH)))
    labels = df[core.KEYS + [core.LABEL_COL]].drop_duplicates()
    summary = core.sales_summary(df).merge(labels, on=core.KEYS)
    top = core.select_top_products(summary, labels=TOP_LABELS, min_total_sales=MIN_TOTAL_SALES)
    print(f"Top products ({'+'.join(TOP_LABELS)}, total sales >= {MIN_TOTAL_SALES}): {len(top)} pairs")
    rows = core.rows_for_pairs(df, top).drop(columns=core.LABEL_COL)
    return _save(rows, out_dir, core.top_products_path(MIN_TOTAL_SALES))


def _save(df, out_dir, default_path):
    path = os.path.join(out_dir, os.path.basename(default_path))
    df.reset_index(drop=True).to_feather(path)
    print(f"  saved -> {path} ({len(df):,} rows, {df.shape[1]} columns)\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    ap.add_argument("--out-dir", default=core.DATASET_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runners = {"fill": step_fill, "classify": step_classify, "top": step_top}
    for step in STEPS:
        if step in args.steps:
            print(f"=== {step} ===")
            runners[step](args.out_dir)


if __name__ == "__main__":
    main()
