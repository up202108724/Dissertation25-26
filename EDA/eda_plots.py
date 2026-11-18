"""
EDA plots: matplotlib for static figures (article), plotly for interactive HTML.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from eda_core import (DATE_COL, ITEM_COL, LABEL_COL, LABEL_ORDER, STORE_COL, VALUE_COL,
                      KEYS, product_rows, promo_columns)


def _new_ax(ax, figsize):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _finish(fig, save_path=None, show=True):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()


def _plot_series(ax, dates, values, title, marker=False):
    ax.plot(dates, values, linewidth=1.5,
            marker="o" if marker else None, markersize=3 if marker else None)
    ax.set_title(title)
    ax.set_ylabel("Sales")
    ax.grid(True, alpha=0.3)


# -----------------------------------------------------------------------------
# Matplotlib
# -----------------------------------------------------------------------------
def plot_label_distribution(labels, order=LABEL_ORDER, ax=None, title="ADI/CV classification",
                            save_path=None, show=True):
    """Bar chart of pairs per ADI/CV label, annotated with count and percentage."""
    counts = labels[LABEL_COL].value_counts().reindex(order, fill_value=0)
    ax = _new_ax(ax, (8, 4))
    bars = ax.bar(counts.index, counts.values)
    total = counts.sum()
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{c} ({c / total:.1%})", ha="center", va="bottom")
    ax.set(title=title, xlabel="Label", ylabel="Number of products")
    _finish(ax.figure, save_path, show)
    return counts


def plot_product(df, item_id, store_id=None, ax=None, marker=False, save_path=None, show=True):
    """Sales of one product over time."""
    p = product_rows(df, item_id, store_id)
    ax = _new_ax(ax, (10, 4))
    _plot_series(ax, p[DATE_COL], p[VALUE_COL], f"Product {item_id}", marker)
    ax.set_xlabel("Date")
    _finish(ax.figure, save_path, show)
    return p


def plot_random_sample_per_label(df, labels, seed=None, order=LABEL_ORDER,
                                 figsize_per_row=(10, 2.6), save_path=None, show=True):
    """One random product per ADI/CV label, stacked with a shared x-axis."""
    rng = np.random.default_rng(seed)
    present = [l for l in order if (labels[LABEL_COL] == l).any()]
    fig, axes = plt.subplots(len(present), 1, sharex=True,
                             figsize=(figsize_per_row[0], figsize_per_row[1] * len(present)))
    axes = np.atleast_1d(axes)
    chosen = {}
    for ax, label in zip(axes, present):
        pool = labels.loc[labels[LABEL_COL] == label, KEYS].to_numpy()
        item_id, store_id = pool[rng.integers(len(pool))]
        p = product_rows(df, item_id, store_id)
        _plot_series(ax, p[DATE_COL], p[VALUE_COL], f"{label.upper()} - Product {item_id}")
        chosen[label] = (item_id, store_id)
    axes[-1].set_xlabel("Date")
    _finish(fig, save_path, show)
    return chosen


def plot_ranked_sales(summary, col="total_sales", title=None, max_labels=50,
                      save_path=None, show=True):
    """Products ranked by ``col`` (e.g. 'total_sales', 'mean_sales')."""
    ranked = summary.sort_values(col, ascending=False).reset_index(drop=True)
    n = len(ranked)
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(range(n), ranked[col], color="skyblue", edgecolor="gray")
    step = 1 if n <= max_labels else max(1, n // 20)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels(ranked[ITEM_COL].iloc[::step].astype(str), rotation=90)
    nice = col.replace("_", " ").title()
    ax.set(title=title or f"{nice} ranked by product", xlabel=f"Product rank (by {nice})",
           ylabel=nice, xlim=(-1, n))
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    _finish(fig, save_path, show)
    return ranked


def plot_sku_share(counts, kind="pie", title=None, save_path=None, show=True):
    """Distinct SKUs per group (output of ``eda_core.skus_per``) as a pie or bar chart."""
    if kind == "pie":
        fig, ax = plt.subplots(figsize=(10, 8))
        counts.plot(kind="pie", autopct="%1.1f%%", startangle=140, ax=ax,
                    colors=plt.cm.Paired.colors, textprops={"fontsize": 11})
        ax.set_ylabel("")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot(kind="bar", color="lightgreen", edgecolor="black", ax=ax)
        ax.set_ylabel("Number of unique SKUs")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_title(title or f"Unique SKUs per {counts.index.name}")
    _finish(fig, save_path, show)


# -----------------------------------------------------------------------------
# Plotly (interactive HTML)
# -----------------------------------------------------------------------------
_RANGE_BUTTONS = [dict(count=1, label="1m", step="month", stepmode="backward"),
                  dict(count=3, label="3m", step="month", stepmode="backward"),
                  dict(count=6, label="6m", step="month", stepmode="backward"),
                  dict(count=1, label="1y", step="year", stepmode="backward"),
                  dict(step="all")]


def plot_product_splits_html(df, item_id, store_id=None, train_size=455, val_size=154,
                             horizon=152, output_dir=None, show=False):
    """Interactive train / validation / test view of one product.

    Splits are taken from the end of the series, as in the baselines:
    test = last ``horizon`` days, validation = the ``val_size`` days before it,
    train = the ``train_size`` days before that.
    """
    import plotly.graph_objects as go

    p = product_rows(df, item_id, store_id)
    test_start = len(p) - horizon
    val_start = test_start - val_size
    train_start = max(0, val_start - train_size)
    parts = {"Train": p.iloc[train_start:val_start],
             "Validation": p.iloc[val_start:test_start],
             "Test": p.iloc[test_start:]}
    fig = go.Figure()
    for (name, part), color in zip(parts.items(), ("blue", "orange", "green")):
        fig.add_trace(go.Scatter(x=part[DATE_COL], y=part[VALUE_COL], mode="lines",
                                 name=name, line=dict(color=color)))
    fig.update_layout(title=f"Item {item_id}" + (f" | Store {store_id}" if store_id else ""),
                      xaxis=dict(title="Date", type="date", rangeslider=dict(visible=True),
                                 rangeselector=dict(buttons=_RANGE_BUTTONS)),
                      yaxis_title="Sales", template="plotly_white")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(os.path.join(output_dir, f"product_{item_id}_ts.html"))
    if show:
        fig.show()
    return fig


def plot_promo_timeline(df_product, title=None, show=True):
    """Sales line with a marker for each day a promotion type is active."""
    import plotly.express as px
    import plotly.graph_objects as go

    p = df_product.sort_values(DATE_COL)
    fig = go.Figure(go.Scatter(x=p[DATE_COL], y=p[VALUE_COL], mode="lines", name="Sales",
                               line=dict(color="rgba(0, 102, 204, 0.7)", width=1.5)))
    colors = px.colors.qualitative.Set1
    active = [c for c in promo_columns(p, "type") if p[c].sum() > 0]
    for i, col in enumerate(active):
        name = col.replace("promo_type_", "")
        days = p[p[col] > 0]
        val_col = col.replace("type", "value")
        promo_val = days[val_col] if val_col in p.columns else [""] * len(days)
        fig.add_trace(go.Scatter(
            x=days[DATE_COL], y=days[VALUE_COL], mode="markers", name=f"{name} active",
            text=[f"Sales: {s}<br>Promo: {name}<br>Value: {v}" for s, v in zip(days[VALUE_COL], promo_val)],
            hovertemplate="%{x|%b %d, %Y}<br>%{text}",
            marker=dict(size=11, symbol="star", color=colors[i % len(colors)],
                        line=dict(width=1, color="black"))))
    if title is None:
        title = f"Sales vs promotions - Item {p[ITEM_COL].iloc[0]} | Store {p[STORE_COL].iloc[0]}"
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Sales (units)",
                      hovermode="x unified", template="plotly_white", height=650,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    if show:
        fig.show()
    return fig
