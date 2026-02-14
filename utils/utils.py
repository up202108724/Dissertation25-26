import pandas as pd
import numpy as np

def select_best_seasonal_smooth(df, top_n=100, max_zero_rate=0.2, sales_col='value', item_col='item_id', label_col='item_label', labels=('seasonal','smooth'), by='total_sales'):
    # filter seasonal / smooth
    df_sub = df[df[label_col].isin(labels)].copy()

    # ensure numeric sales
    df_sub[sales_col] = pd.to_numeric(df_sub[sales_col], errors='coerce').fillna(0)

    # compute metrics per item
    agg = df_sub.groupby(item_col).agg(
        days_observed=('date', 'nunique'),
        total_sales=(sales_col, 'sum'),
        mean_sales=(sales_col, 'mean'),
        median_sales=(sales_col, 'median'),
        zeros=(sales_col, lambda x: (x==0).sum()),
        obs_count=(sales_col, 'count')
    ).reset_index()

    agg['zero_rate'] = agg['zeros'] / agg['obs_count']
    agg['nonzero_mean'] = agg.apply(lambda r: r['total_sales'] / max(1, r['obs_count'] - r['zeros']), axis=1)

    # filter by zero rate
    agg = agg[agg['zero_rate'] <= max_zero_rate]

    # choose sort key
    if by == 'total_sales':
        agg = agg.sort_values('total_sales', ascending=False)
    elif by == 'mean_sales':
        agg = agg.sort_values('mean_sales', ascending=False)
    elif by == 'nonzero_mean':
        agg = agg.sort_values('nonzero_mean', ascending=False)
    else:
        raise ValueError("by must be 'total_sales', 'mean_sales' or 'nonzero_mean'")

    # take top_n items and return original rows for those items
    top_items = agg.head(top_n)[item_col].tolist()
    return df_sub[df_sub[item_col].isin(top_items)].copy(), agg.loc[agg[item_col].isin(top_items)]

# Example usage:
# subset_rows, top_summary = select_best_seasonal_smooth(df_classified, top_n=50, max_zero_rate=0.15)