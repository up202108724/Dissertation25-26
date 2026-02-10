from plotly.subplots import make_subplots
import plotly.graph_objects as go

def analyze_product_timeseries(item_id, df_classified):
  
    product_data = df_classified[df_classified['item_id'] == item_id].sort_index()
    
    if len(product_data) == 0:
        print(f"⚠️ Item ID {item_id} not found in dataset")
        return
    
    label = product_data['item_label'].iloc[0]
    dep = product_data['dep_label'].iloc[0] if 'dep_label' in product_data.columns else "N/A"

    print(f"\n{'='*70}")
    print(f"Product ID: {item_id}")
    print(f"Department: {dep}")
    print(f"Classification: {label.upper()}")
    print(f"{'='*70}")

    print(f"\nTime Series Statistics:")
    print(f"  Period: {product_data.index.min().date()} to {product_data.index.max().date()}")
    print(f"  Total records: {len(product_data)}")
    print(f"  Non-zero sales: {(product_data['value'] > 0).sum()}")
    print(f"  Zero-sales periods: {(product_data['value'] == 0).sum()}")
    print(f"\n  Sales Stats:")
    print(f"    Mean: {product_data['value'].mean():.2f}")
    print(f"    Median: {product_data['value'].median():.2f}")
    print(f"    Min: {product_data['value'].min():.2f}")
    print(f"    Max: {product_data['value'].max():.2f}")
    print(f"    Std Dev: {product_data['value'].std():.2f}")
    
    # Plot time series
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=product_data.index,
            y=product_data['value'],
            mode='lines+markers',
            name=f'Item {item_id}',
            line=dict(color='#3498db', width=2),
            marker=dict(size=6)
        )
    )
    
    # Color code zero vs non-zero sales
    zero_sales = product_data[product_data['value'] == 0]
    if len(zero_sales) > 0:
        fig.add_trace(
            go.Scatter(
                x=zero_sales.index,
                y=zero_sales['value'],
                mode='markers',
                name='Zero Sales',
                marker=dict(color='red', size=8, symbol='x'),
                showlegend=True
            )
        )
    
    fig.update_layout(
        title=f"Time Series: Item {item_id} ({label.upper()})",
        xaxis_title="Date",
        yaxis_title="Sales Value",
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.show()


def search_products(df_classified, label=None, department=None, item_id=None):
    """
    Search for products by classification label, department, or item ID.
    
    Parameters:
    label : str - Filter by classification ('regular', 'seasonal', 'intermittent', 'new')
    department : str - Filter by department name
    item_id : int - Exact item ID to find
    
    Returns:
    DataFrame with matching products
    """
    result = df_classified.copy()
    
    if item_id is not None:
        result = result[result['item_id'] == item_id]
    
    if label is not None:
        result = result[result['item_label'] == label.lower()]
    
    if department is not None:
        result = result[result['dep_label'].str.contains(department, case=False, na=False)]
    
    # Get unique items
    unique_items = result[['item_id', 'item_label', 'dep_label']].drop_duplicates()
    
    print(f"Found {len(unique_items)} items matching criteria:")
    print(f"\nClassification breakdown:")
    print(unique_items['item_label'].value_counts())
    
    return unique_items


# Example searches:
print("SEARCH EXAMPLES:")
print("\n1. Find all seasonal items:")
print("   seasonal_items = search_products(label='seasonal')")
print("\n2. Find all items in a department:")
print("   dept_items = search_products(department='Electronics')")
print("\n3. Find a specific product:")
print("   product = search_products(item_id=12345)")
print("\n4. Analyze a product:")
print("   analyze_product_timeseries(12345)")