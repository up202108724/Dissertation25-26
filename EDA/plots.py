import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def plot_label_distribution(
    label_counts,
    ax=None,
    figsize=(8, 4),
    title="Item Classification Distribution",
    xlabel="Classification Label",
    ylabel="Count",
    annotate=True,
    sort=None,          # None | "asc" | "desc"
    order=None,         # e.g. ["regular","seasonal","intermittent","new"]
    print_breakdown=True,
    show=True
):
    # Normalize to plain dict
    if hasattr(label_counts, "to_dict"):
        label_counts = label_counts.to_dict()
    else:
        label_counts = dict(label_counts)

    # Build label/count lists
    items = list(label_counts.items())

    # Apply explicit order if given
    if order is not None:
        order_index = {lab: i for i, lab in enumerate(order)}
        items.sort(key=lambda kv: order_index.get(kv[0], len(order) + 1))
    elif sort in ("asc", "desc"):
        items.sort(key=lambda kv: kv[1], reverse=(sort == "desc"))

    labels = [k for k, _ in items]
    counts = [v for _, v in items]

    total = sum(counts) if counts else 0
    breakdown = {
        lab: {"count": cnt, "percentage": (cnt / total * 100 if total else 0.0)}
        for lab, cnt in zip(labels, counts)
    }

    # Create axis if not provided
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # Plot
    bars = ax.bar(labels, counts)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Annotate
    if annotate:
        for b, c in zip(bars, counts):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                str(c),
                ha="center",
                va="bottom"
            )

    if created_fig:
        plt.tight_layout()

    # Print breakdown
    if print_breakdown:
        print("\nDetailed Classification Distribution:")
        print(f"Total items: {total}")
        print("\nBreakdown:")
        for lab in labels:
            cnt = breakdown[lab]["count"]
            pct = breakdown[lab]["percentage"]
            print(f"  {lab:12s}: {cnt:6d} items ({pct:5.1f}%)")

    if show:
        plt.show()


def plot_product_value_evolution(
    df,
    selected_product=None,
    item_col="item_id",
    date_col=None,      # column name OR None/"index" to use DatetimeIndex
    value_col="value",
    ax=None,
    figsize=(10, 4),
    title_prefix="Sales Value Evolution",
    marker=True,
    print_stats=True,
    agg_per_date="sum",   # None | "sum" | "mean"
    show=True,
):
    """
    Plot a single product's sales value over time using matplotlib.

    date_col can be:
      - a column name (e.g., "date")
      - None or "index" meaning: use df.index (must be DatetimeIndex)

    Returns
    -------
    ax : matplotlib.axes.Axes
    product_data : pd.DataFrame with columns ["date", value_col]
    stats : dict
    """
    if selected_product is None:
        selected_product = df[item_col].iloc[0]

    dfi = df.loc[df[item_col] == selected_product].copy()

    # ---- get dates from either a column or the index
    if date_col is None or date_col == "index":
        if not isinstance(dfi.index, pd.DatetimeIndex):
            raise ValueError("date_col is None/'index' but df.index is not a DatetimeIndex.")
        dates = pd.to_datetime(dfi.index)
    else:
        if date_col not in dfi.columns:
            raise KeyError(f"'{date_col}' not found in df columns. Use date_col=None or 'index' to use DatetimeIndex.")
        dates = pd.to_datetime(dfi[date_col])

    # Build a tidy frame with explicit date column
    product_data = pd.DataFrame({
        "date": dates,
        value_col: dfi[value_col].values
    }).sort_values("date")

    # Optional aggregation per date (if duplicates)
    if agg_per_date in ("sum", "mean"):
        func = "sum" if agg_per_date == "sum" else "mean"
        product_data = product_data.groupby("date", as_index=False)[value_col].agg(func)

    stats = {
        "product_id": selected_product,
        "date_min": product_data["date"].min(),
        "date_max": product_data["date"].max(),
        "total_records": len(product_data),
        "non_zero_sales": int((product_data[value_col] > 0).sum()),
        "describe": product_data[value_col].describe(),
    }

    if print_stats:
        print(f"Product ID: {stats['product_id']}")
        print(f"Date range: {stats['date_min']} to {stats['date_max']}")
        print(f"Total records: {stats['total_records']}")
        print(f"Non-zero sales: {stats['non_zero_sales']}")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    ax.plot(
        product_data["date"],
        product_data[value_col],
        linewidth=2,
        marker="o" if marker else None,
        markersize=4 if marker else None,
    )

    ax.set_title(f"{title_prefix} - Product {selected_product}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales Value")
    ax.grid(True, alpha=0.3)

    if created_fig:
        plt.tight_layout()
    if show:
        plt.show()

    return ax, product_data, stats

def plot_random_sample_per_label(
    df,
    df_classified,
    item_col="item_id",
    date_col="date",
    value_col="value",
    label_col="item_label",
    seed=None,                 # <- default None = different every call
    rng=None,                  # <- pass an RNG for evolving samples across calls
    figsize_per_row=(10, 3),
    marker=True,
    print_stats=False,
    agg_per_date="sum",
    show=True,
):
    """
    Plot one random sample of each classification label.

    If you pass `seed` (int), results are deterministic for that call.
    If you pass `rng` (np.random.Generator), sampling will differ across calls as rng state advances.
    If both are None, uses entropy for different results each call.
    
    Parameters
    ----------
    date_col : str or None
        If None, date is assumed to be the index (DatetimeIndex).
        If str, date is a column name in df.
    """
    if rng is None:
        rng = np.random.default_rng(seed)  # seed=None => entropy

    labels = df_classified[label_col].dropna().unique().tolist()
    labels = sorted(labels)

    n_rows = len(labels)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows),
        sharex=True
    )
    if n_rows == 1:
        axes = [axes]

    results = {}

    for ax, label in zip(axes, labels):
        items_with_label = df_classified.loc[df_classified[label_col] == label, item_col].dropna().unique()

        if len(items_with_label) == 0:
            ax.text(0.5, 0.5, f"No items with label: {label}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label.upper())
            ax.grid(True, alpha=0.3)
            continue

        selected_product = rng.choice(items_with_label)

        # Handle date_col=None (date is the index)
        if date_col is None:
            product_data = df.loc[df[item_col] == selected_product, value_col].sort_index()
            product_data = product_data.to_frame()
            product_data.columns = [value_col]
            dates = product_data.index
        else:
            product_data = df.loc[df[item_col] == selected_product, [date_col, value_col]].copy()
            product_data[date_col] = pd.to_datetime(product_data[date_col])
            product_data = product_data.sort_values(date_col)
            dates = product_data[date_col]

        if agg_per_date in ("sum", "mean"):
            func = "sum" if agg_per_date == "sum" else "mean"
            if date_col is None:
                product_data = product_data.groupby(level=0)[value_col].agg(func).to_frame()
            else:
                product_data = product_data.groupby(date_col, as_index=False)[value_col].agg(func)

        # Recalculate dates after aggregation
        if date_col is None:
            dates = product_data.index
        else:
            dates = product_data[date_col]

        stats = {
            "label": label,
            "product_id": selected_product,
            "date_min": dates.min(),
            "date_max": dates.max(),
            "total_records": len(product_data),
            "non_zero_sales": int((product_data[value_col] > 0).sum()),
        }
        results[label] = stats

        if print_stats:
            print(f"\n{label.upper()} - Product {selected_product}")
            print(f"  Date range: {stats['date_min']} to {stats['date_max']}")
            print(f"  Total records: {stats['total_records']}")
            print(f"  Non-zero sales: {stats['non_zero_sales']}")

        ax.plot(
            dates,
            product_data[value_col].values,
            linewidth=2,
            marker="o" if marker else None,
            markersize=4 if marker else None,
        )
        ax.set_title(f"{label.upper()} - Product {selected_product}")
        ax.set_ylabel("Sales Value")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes, results
