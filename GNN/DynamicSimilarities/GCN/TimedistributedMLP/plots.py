import pandas as pd
import numpy as np
import networkx as nx
import plotly.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def plot_results(train, val, test, forecast, 
                train_index, val_index, test_index, 
                train_losses, val_losses, metric=None, embedding_strategy=None,
                window_size=None, step_size=None, threshold=None, percentile=None,
                enable_edges_within_star=None, seed=None,
                target_col='value', title='Forecast vs Actual', save_path=None,
                rmse=None, mae=None, bias=None, score=None, pocid=None, df_full=None):
    
    # Check if forecast is a dictionary; if not, wrap it in a dictionary for uniformity
    if not isinstance(forecast, dict):
        forecast = {'Forecast': forecast}
        rmse = {'Forecast': rmse}
        mae = {'Forecast': mae}
        bias = {'Forecast': bias}
        score = {'Forecast': score}
        pocid = {'Forecast': pocid}
        train_losses = {'Forecast': train_losses}
        val_losses = {'Forecast': val_losses}
        
    # Calculate metrics if not provided
    for label, fcast in forecast.items():
        if fcast is None:
            rmse[label], mae[label], bias[label], score[label] = 0, 0, 0, 0
            continue
            
        if rmse.get(label) is None:
            fcast_array = np.array(fcast, dtype=float)
            valid_mask = ~np.isnan(fcast_array)
            valid_test = test[valid_mask]
            valid_forecast = fcast_array[valid_mask]
            if len(valid_test) > 0:
                rmse[label] = np.sqrt(mean_squared_error(valid_test, valid_forecast))
                mae[label] = mean_absolute_error(valid_test, valid_forecast)
                bias[label] = np.mean(valid_forecast - valid_test)
                score[label] = r2_score(valid_test, valid_forecast)
                # pocid would need to be calculated here ideally, but logic isn't provided here yet
            else:
                rmse[label], mae[label], bias[label], score[label] = 0, 0, 0, 0
            
    # Remove large individual metrics from title - instead we'll place them in the legend or skip them if multiple
    is_multi = len(forecast) > 1
        
    # Update title conditionally
    meta_parts = []
    if embedding_strategy is not None:
        meta_parts.append(f"Graph Params: {embedding_strategy}")
    if metric is not None:
        meta_parts.append(f"Metric: {metric}")
    if window_size is not None:
        meta_parts.append(f"Window: {window_size}")
    if step_size is not None:
        meta_parts.append(f"Step: {step_size}")
    if threshold is not None:
        meta_parts.append(f"Threshold: {threshold}")
    if percentile is not None:
        meta_parts.append(f"Percentile: {percentile}")
    if enable_edges_within_star is not None:
        meta_parts.append(f"Edges in Star: {enable_edges_within_star}")
    if seed is not None:
        meta_parts.append(f"Seed: {seed}")
    metadata = " | ".join(meta_parts)
    meta_html = f"<span style='font-size:14px;color:gray'>{metadata}</span><br>" if metadata else ""
    
    full_title = f"{title}<br>{meta_html}"
    if not is_multi:
        label = list(forecast.keys())[0]
        pocid_str = f"{pocid[label]:.4f}" if pocid.get(label) is not None else "N/A"
        full_title += f"RMSE: {rmse[label]:.4f} | MAE: {mae[label]:.4f} | Bias: {bias[label]:.4f} | Score: {score[label]:.4f} | POCID: {pocid_str}"
        
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=(full_title, 'Test vs Forecast', 'Training and Validation Loss'),
                        vertical_spacing=0.08,
                        row_heights=[0.45, 0.30, 0.25])
    
    # Define a common hovertemplate to include date and value
    hover_temp = 'Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}'

    # Plot forecast vs actual - ax1
    fig.add_trace(go.Scatter(x=train_index, y=train, name='Train', opacity=0.7, mode='lines', hovertemplate=hover_temp), row=1, col=1)
    fig.add_trace(go.Scatter(x=val_index, y=val, name='Validation', opacity=0.7, mode='lines', line=dict(color='orange'), hovertemplate=hover_temp), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_index, y=test, name='Actual Test', mode='lines', line=dict(color='green', width=2), hovertemplate=hover_temp), row=1, col=1)
    
    import plotly.express as px
    colors = px.colors.qualitative.Plotly
    
    for idx, (label, fcast) in enumerate(forecast.items()):
        if fcast is None: continue
        color = colors[idx % len(colors)]
        if not is_multi:
            color = 'red'
        
        legend_name = label
        if is_multi:
             pocid_str  = f"{pocid.get(label):.4f}"  if pocid  and pocid.get(label)  is not None else "N/A"
             bias_str   = f"{bias.get(label):.4f}"   if bias   and bias.get(label)   is not None else "N/A"
             score_str  = f"{score.get(label):.4f}"  if score  and score.get(label)  is not None else "N/A"
             legend_name = (
                 f"{label} "
                 f"(RMSE: {rmse[label]:.2f} | MAE: {mae[label]:.2f} | "
                 f"Bias: {bias_str} | Score: {score_str} | POCID: {pocid_str})"
             )
             
        fig.add_trace(go.Scatter(x=test_index, y=fcast, name=legend_name, legendgroup=label, mode='lines', line=dict(color=color, width=2), hovertemplate=hover_temp), row=1, col=1)
    
    
    # Pinpoint specific dates
    fig.add_vline(x='2022-09-24', line_dash="dot", line_color="purple", line_width=2, row=1, col=1)
    if len(test_index) > 0:
        # Use .iloc[-1] to access the last element by position, not by label
        if hasattr(test_index, 'iloc'):
            last_date = test_index.iloc[-1]
        else:
            last_date = test_index[-1]
        fig.add_vline(x=last_date, line_dash="dot", line_color="brown", line_width=2, row=1, col=1)
        
    # Mark promotion days using scatter points
    if df_full is not None:
        # Detect promo columns dynamically
        promo_type_cols = [c for c in df_full.columns if c.startswith('promo_type_')]
        promo_colors = ['rgba(255, 0, 255, 0.5)', 'rgba(0, 255, 255, 0.5)', 'rgba(255, 255, 0, 0.5)']
        color_idx = 0
        
        for promo_col in promo_type_cols:
            promo_dates = df_full[df_full[promo_col] == 1]['date']
            # Find the target values for these dates across train, val, and test to correctly place markers
            y_vals = []
            valid_dates = []
            
            for d in promo_dates:
                # Combine train, val, test arrays and indices
                all_idx = np.concatenate([train_index, val_index, test_index])
                all_y = np.concatenate([train, val, test])
                
                # Check if date is in indices
                if d in all_idx:
                    idx_pos = np.where(all_idx == d)[0][0]
                    y_vals.append(all_y[idx_pos])
                    valid_dates.append(d)
                    
            if valid_dates:
                 color = promo_colors[color_idx % len(promo_colors)]
                 promo_name = promo_col.replace('promo_type_', '')
                 # Add scatter points where promotions occur
                 fig.add_trace(go.Scatter(x=valid_dates, y=y_vals, mode='markers',
                                          name=f'Promo: {promo_name}',
                                          marker=dict(color=color, size=10, symbol='star')), 
                               row=1, col=1)
                 color_idx += 1
                 
    # Set tick format for x-axis to 3 months     
    fig.update_xaxes(
        title_text='Date', 
        dtick="M3", 
        tickformat="%b\n%Y",
        row=1, col=1
    )
    fig.update_yaxes(title_text=target_col, row=1, col=1)
    
    # Plot forecast vs actual test only - ax2
    fig.add_trace(go.Scatter(x=test_index, y=test, name='Actual Test (Zoom)', mode='lines', line=dict(color='green', width=2), showlegend=False, hovertemplate=hover_temp), row=2, col=1)
    
    for idx, (label, fcast) in enumerate(forecast.items()):
        if fcast is None: continue
        color = colors[idx % len(colors)]
        if not is_multi:
            color = 'red'
        fig.add_trace(go.Scatter(x=test_index, y=fcast, name=label + ' (Zoom)', legendgroup=label, mode='lines', line=dict(color=color, width=2), showlegend=False, hovertemplate=hover_temp), row=2, col=1)
    
    fig.update_xaxes(
        title_text='Date',
        dtick="M3",
        tickformat="%b\n%Y",
        row=2, col=1
    )
    fig.update_yaxes(title_text=target_col, row=2, col=1)

    # Plot training loss - ax3
    if not isinstance(train_losses, dict):
        train_losses = {'Forecast': train_losses}
        val_losses = {'Forecast': val_losses}
        
    for idx, (label, t_loss) in enumerate(train_losses.items()):
        if t_loss is None:
            continue
        v_loss = val_losses.get(label, [])
        color = colors[idx % len(colors)]
        epochs = list(range(1, len(t_loss) + 1))
        
        name_t = 'Train Loss' if not is_multi else f'Train Loss ({label})'
        name_v = 'Validation Loss' if not is_multi else f'Val Loss ({label})'
        
        fig.add_trace(go.Scatter(x=epochs, y=t_loss, name=name_t, legendgroup=label, mode='lines', line=dict(color=color, dash='solid')), row=3, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=v_loss, name=name_v, legendgroup=label, mode='lines', line=dict(color=color, dash='dot')), row=3, col=1)
    
    fig.update_yaxes(title_text='Loss', row=3, col=1)
    fig.update_xaxes(title_text='Epoch', row=3, col=1)
    
    fig.update_layout(height=1400, width=1800,
                      hovermode="x unified",
                      template="plotly_white",
                      margin=dict(l=60, r=40, t=120, b=60),
                      legend=dict(orientation="v", yanchor="top", y=1.0,
                                  xanchor="left", x=1.02))
    
    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            # If kaleido is installed, it can save to png/jpg/pdf. 
            # Otherwise we'll fallback to writing html if it fails.
            try:
                fig.write_image(save_path)
            except Exception:
                fig.write_html(save_path + '.html')
    else:
        fig.show()
        
def plot_networkx_plotly(G, title="Network Graph", save_path=None, target_node=None):
    """
    Plots a NetworkX graph using Plotly.
    Calculates the graph layout using spring_layout and creates interactive hovering.
    """
    # 1. Get Node Positions
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # 2. Add edges and edge hover points to Plotly traces
    edge_x = []
    edge_y = []
    edge_mid_x = []
    edge_mid_y = []
    edge_text = []
    
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        edge_mid_x.append((x0 + x1) / 2)
        edge_mid_y.append((y0 + y1) / 2)
        
        weight = data.get('weight', 'N/A')
        sim = data.get('similarity', None)
        val_str = f"Weight (Dist): {weight:.4f}" if isinstance(weight, float) else f"Weight: {weight}"
        if sim is not None:
            val_str += f"<br>Similarity: {sim:.4f}" if isinstance(sim, float) else f"<br>Similarity: {sim}"
            
        edge_text.append(f"{u} - {v}<br>{val_str}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    edge_hover_trace = go.Scatter(
        x=edge_mid_x, y=edge_mid_y,
        mode='markers',
        marker=dict(size=10, color='rgba(0,0,0,0)'), # Invisible maker but large enough to hover easily
        hoverinfo='text',
        text=edge_text,
        showlegend=False
    )

    # 3. Add nodes to Plotly grouped by category
    categories = {}
    for node_id, data in G.nodes(data=True):
        cat = data.get('cat_label', 'Unknown Category')
        if cat not in categories:
            categories[cat] = {
                'x': [], 'y': [], 'sizes': [], 'lines_w': [], 
                'lines_c': [], 'text': [], 'hovertext': []
            }
        
        x, y = pos[node_id]
        degree = len(list(G.neighbors(node_id)))
        
        categories[cat]['x'].append(x)
        categories[cat]['y'].append(y)
        categories[cat]['text'].append(str(node_id))
        
        if target_node is not None and node_id == target_node:
            categories[cat]['hovertext'].append(f'🎯 TARGET NODE: {node_id}<br>Category: {cat}<br>Connections: {degree}')
            categories[cat]['sizes'].append(35)
            categories[cat]['lines_w'].append(3)
            categories[cat]['lines_c'].append('red')
        else:
            categories[cat]['hovertext'].append(f'Node: {node_id}<br>Category: {cat}<br>Connections: {degree}')
            categories[cat]['sizes'].append(20)  # Larger so text fits
            categories[cat]['lines_w'].append(1)
            categories[cat]['lines_c'].append('black')

    traces = [edge_trace, edge_hover_trace]
    palette = pcolors.qualitative.Plotly
    
    import hashlib
    for cat, d in categories.items():
        # Use a deterministic hash to ensure consistent color across different graphs and Python runs
        color_idx = int(hashlib.md5(str(cat).encode()).hexdigest(), 16) % len(palette)
        color = palette[color_idx]
        
        node_trace = go.Scatter(
            x=d['x'], y=d['y'],
            mode='markers+text',
            hoverinfo='text',
            text=d['text'],
            hovertext=d['hovertext'],
            textposition="top center",
            textfont=dict(size=12, color='black'),
            marker=dict(
                color=color,
                size=d['sizes'],
                line=dict(width=d['lines_w'], color=d['lines_c'])
            ),
            name=str(cat)
        )
        traces.append(node_trace)

    # 4. Create final figure layout
    fig = go.Figure(data=traces,
             layout=go.Layout(
                title=dict(text=title, font=dict(size=16)),
                showlegend=True,  # Show category legend
                legend=dict(title=dict(text='Categories')),
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
             )
    )
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()
