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
                rmse=None, mae=None, bias=None, score=None, pocid=None, df_full=None,
                inference_step_dates=None, neighbour_series=None,
                inference_step_neighbours=None, all_step_neighbours=None,
                neighbour_counts=None):
    
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

    # ── Optional 3rd panel: graph-neighbourhood size per forecast step ────────
    # Only rendered when the caller supplies per-label neighbour counts (the
    # GCN/GAT/Graph2Vec runner).  Baselines / ablations contribute no counts, so
    # the panel stays focused on the graph variants.
    show_nbr_panel = bool(neighbour_counts) and any(
        v is not None and len(v) > 0 for v in neighbour_counts.values()
    )
    if show_nbr_panel:
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(full_title, 'Training and Validation Loss',
                            'Graph neighbourhood size per forecast step'),
            vertical_spacing=0.08,
            row_heights=[0.58, 0.24, 0.18],
        )
    else:
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=(full_title, 'Training and Validation Loss'),
                            vertical_spacing=0.10,
                            row_heights=[0.70, 0.30])
    
    # Define a common hovertemplate to include date and value
    hover_temp = 'Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}'
    step_indices = np.arange(len(test_index))

    _hover_plain = 'Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<br>Step: %{customdata}'
    _hover_nbrs  = (
        'Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}'
        '<br>Step: %{customdata[0]}'
        '<br>Neighbours: %{customdata[1]}<extra></extra>'
    )

    def _trace_customdata(label):
        """Return (customdata, hovertemplate) with threshold-specific neighbours."""
        if all_step_neighbours is None:
            return step_indices, _hover_plain
        # Ablation traces use zeroed GCN embeddings — no meaningful graph neighbours
        if 'az:ablation' in label:
            return step_indices, _hover_plain
        th_key = None
        if 'th:' in label:
            try:
                th_key = float(label.split('th:')[1].split('|')[0])
            except (IndexError, ValueError):
                pass
        step_nbrs = all_step_neighbours.get(th_key) if th_key is not None else None
        if step_nbrs is None:
            return step_indices, _hover_plain
        nbr_strs = [
            ', '.join(str(n) for n in step_nbrs.get(i, [])) or '—'
            for i in range(len(test_index))
        ]
        cd = np.array([[i, s] for i, s in zip(step_indices, nbr_strs)], dtype=object)
        return cd, _hover_nbrs

    # Plot forecast vs actual - ax1
    fig.add_trace(go.Scatter(x=train_index, y=train, name='Train', opacity=0.7, mode='lines', hovertemplate=hover_temp), row=1, col=1)
    fig.add_trace(go.Scatter(x=val_index, y=val, name='Validation', opacity=0.7, mode='lines', line=dict(color='orange'), hovertemplate=hover_temp), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_index, y=test, name='Actual Test', mode='lines', line=dict(color='green', width=2), customdata=step_indices, hovertemplate=_hover_plain), row=1, col=1)

    import plotly.express as px
    colors = px.colors.qualitative.Plotly

    # Shared label→colour map so the neighbour panel (row 3) and the loss panel
    # (row 2) reuse exactly the colour each forecast line gets in row 1.
    label_color = {
        label: ('red' if not is_multi else colors[idx % len(colors)])
        for idx, label in enumerate(forecast.keys())
    }

    for idx, (label, fcast) in enumerate(forecast.items()):
        if fcast is None: continue
        color = label_color[label]

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

        cd, ht = _trace_customdata(label)
        fig.add_trace(go.Scatter(x=test_index, y=fcast, name=legend_name, legendgroup=label, mode='lines', line=dict(color=color, width=2), customdata=cd, hovertemplate=ht), row=1, col=1)
    
    
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
                 
    # ── Neighbour time series (ego-graph nodes at selected inference steps) ──
    if neighbour_series is not None and len(neighbour_series) > 0:
        test_arr = np.array(test, dtype=float)
        t_min, t_max = np.nanmin(test_arr), np.nanmax(test_arr)
        t_range = t_max - t_min if t_max > t_min else 1.0
        nbr_palette = pcolors.qualitative.Pastel + pcolors.qualitative.Light24
        for i, (nbr_id, nbr_vals) in enumerate(neighbour_series.items()):
            nbr_vals = np.array(nbr_vals, dtype=float)
            # Align length to test_index (truncate or pad with NaN)
            n_exp = len(test_index)
            if len(nbr_vals) < n_exp:
                nbr_vals = np.concatenate([nbr_vals, np.full(n_exp - len(nbr_vals), np.nan)])
            else:
                nbr_vals = nbr_vals[:n_exp]
            # Normalise to the target test range so shapes are comparable
            n_min, n_max = np.nanmin(nbr_vals), np.nanmax(nbr_vals)
            n_range = n_max - n_min if n_max > n_min else 1.0
            nbr_scaled = (nbr_vals - n_min) / n_range * t_range + t_min
            color = nbr_palette[i % len(nbr_palette)]
            hover = f'Neighbour: {nbr_id}<br>Date: %{{x|%Y-%m-%d}}<br>Normalised: %{{y:.2f}}'
            shared_kwargs = dict(
                x=test_index, y=nbr_scaled,
                mode='lines',
                line=dict(color=color, width=1),
                hovertemplate=hover,
            )
            fig.add_trace(go.Scatter(**shared_kwargs, name=f'Nbr {nbr_id}'), row=1, col=1)

    # ── Red vlines at selected inference steps ────────────────────────────
    if inference_step_dates is not None:
        test_arr = np.array(test, dtype=float)
        _y_mid = float(np.nanmean(test_arr))
        for step_date in inference_step_dates:
            x_val = pd.Timestamp(step_date).isoformat()
            fig.add_vline(x=x_val, line_dash='solid', line_color='red',
                          line_width=1.5, opacity=0.6, row=1, col=1)
            # Invisible marker on the vline so hovering shows neighbour info
            if inference_step_neighbours is not None:
                nbrs = inference_step_neighbours.get(step_date, [])
                nbr_text = ', '.join(str(n) for n in nbrs) if nbrs else 'none'
                hover_txt = (
                    f"<b>Inference step</b>: {pd.Timestamp(step_date).date()}"
                    f"<br><b>Neighbours ({len(nbrs)})</b>: {nbr_text}"
                )
                fig.add_trace(
                    go.Scatter(
                        x=[pd.Timestamp(step_date)],
                        y=[_y_mid],
                        mode='markers',
                        marker=dict(size=12, opacity=0.0, color='red',
                                    symbol='diamond', line=dict(width=1, color='darkred')),
                        hovertemplate=hover_txt + '<extra></extra>',
                        name='Graph neighbours',
                        legendgroup='graph_neighbours',
                        showlegend=False,
                    ),
                    row=1, col=1,
                )

    # Set tick format for x-axis to 3 months
    fig.update_xaxes(
        title_text='Date',
        dtick="M3",
        tickformat="%b\n%Y",
        row=1, col=1
    )
    fig.update_yaxes(title_text=target_col, row=1, col=1)
    
    # Plot training loss - ax2
    if not isinstance(train_losses, dict):
        train_losses = {'Forecast': train_losses}
        val_losses = {'Forecast': val_losses}
        
    for idx, (label, t_loss) in enumerate(train_losses.items()):
        if t_loss is None:
            continue
        v_loss = val_losses.get(label, [])
        color = label_color.get(label, colors[idx % len(colors)])
        epochs = list(range(1, len(t_loss) + 1))

        name_t = 'Train Loss' if not is_multi else f'Train Loss ({label})'
        name_v = 'Validation Loss' if not is_multi else f'Val Loss ({label})'

        fig.add_trace(go.Scatter(x=epochs, y=t_loss, name=name_t, legendgroup=label, mode='lines', line=dict(color=color, dash='solid')), row=2, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=v_loss, name=name_v, legendgroup=label, mode='lines', line=dict(color=color, dash='dot')), row=2, col=1)

    fig.update_yaxes(title_text='Loss', row=2, col=1)
    fig.update_xaxes(title_text='Epoch', row=2, col=1)

    # ── Row 3: graph-neighbourhood size per forecast step ─────────────────────
    # One line per graph variant, coloured to match its forecast line.  The x
    # axis is the forecast (test) date so spikes line up with the demand series
    # in row 1 — e.g. a neighbourhood that grows around a retail event.
    if show_nbr_panel:
        n_steps = len(test_index)
        for label, counts in neighbour_counts.items():
            if counts is None or len(counts) == 0:
                continue
            counts = list(counts)
            # Align to the forecast horizon (truncate / NaN-pad) so x and y match.
            if len(counts) < n_steps:
                counts = counts + [np.nan] * (n_steps - len(counts))
            else:
                counts = counts[:n_steps]
            color = label_color.get(label, '#1f77b4')
            fig.add_trace(
                go.Scatter(
                    x=test_index, y=counts, name=f'Neighbours ({label})',
                    legendgroup=label, showlegend=False, mode='lines+markers',
                    line=dict(color=color, width=1.5), marker=dict(size=4),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Neighbours: %{y}<extra></extra>',
                ),
                row=3, col=1,
            )
        fig.update_yaxes(title_text='# neighbours', rangemode='tozero', row=3, col=1)
        fig.update_xaxes(title_text='Date', dtick="M3", tickformat="%b\n%Y", row=3, col=1)

    fig.update_layout(height=1750 if show_nbr_panel else 1400, width=1800,
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


def plot_threshold_network(matrix, node_labels, threshold, metric_type='similarity', title=None):
    """
    Plots an interactive network graph using Plotly and NetworkX.
    Nodes with 0 connections are hidden from the plot.
    
    Args:
        matrix (np.ndarray): 2D array of similarities or distances (NxN).
        node_labels (list): List of product IDs or names for the nodes.
        threshold (float): The cutoff value for creating an edge.
        metric_type (str): 'similarity' (keep >= threshold) or 'distance' (keep <= threshold).
        title (str): Custom title for the plot.
        
    Returns:
        fig: A Plotly graph object figure.
    """
    if title is None:
        direction = ">=" if metric_type == 'similarity' else "<="
        title = f"Product Network ({metric_type.capitalize()} {direction} {threshold})"

    # 1. Initialize the NetworkX Graph
    G = nx.Graph()
    
    # Add nodes
    for i, label in enumerate(node_labels):
        G.add_node(i, label=str(label))
        
    # 2. Add edges based on the threshold rule
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(i + 1, cols):  # Upper triangle to avoid duplicate edges
            val = matrix[i, j]
            # Ignore self-loops and NaN values
            if i == j or np.isnan(val):
                continue
                
            if metric_type == 'similarity' and val >= threshold:
                G.add_edge(i, j, weight=val)
            elif metric_type == 'distance' and val <= threshold:
                G.add_edge(i, j, weight=val)

    # ---------------------------------------------------------
    # 2.5 NEW: Remove isolated nodes (nodes with 0 connections)
    # ---------------------------------------------------------
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    # If the graph becomes completely empty, handle gracefully
    if len(G.nodes()) == 0:
        print(f"Warning: No connections found for threshold {threshold}. Graph is empty.")
        # Return an empty figure with a warning title
        return go.Figure(layout=go.Layout(title=f"No connections found at {direction} {threshold}"))

    # 3. Calculate Layout (Spring layout spaces out highly connected nodes nicely)
    pos = nx.spring_layout(G, seed=42)

    # 4. Create Plotly Edge Trace
    edge_x = []
    edge_y = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.0, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # 5. Create Plotly Node Trace
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]['label'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=15,
                title=dict(text='Node Degree', side='right'),
                xanchor='left'
            ),
            line_width=2
        )
    )

    # 6. Color nodes by their degree (number of connections)
    # 6. Color nodes by their degree (number of connections)
    node_adjacencies = []
    node_hovertext = []
    
    # FIX: Remove enumerate. G.adjacency() yields (node_id, adjacency_dict)
    for node, adj_dict in G.adjacency():
        degree = len(adj_dict)
        node_adjacencies.append(degree)
        node_hovertext.append(f"Product: {G.nodes[node]['label']}<br>Connections: {degree}")

    node_trace.marker.color = node_adjacencies
    node_trace.hovertext = node_hovertext

    node_trace.marker.color = node_adjacencies
    node_trace.hovertext = node_hovertext

    # 7. Assemble the Figure
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(text=title, font=dict(size=16)),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[dict(
                    text=f"Total Nodes (Connected): {len(G.nodes())} | Total Edges: {len(G.edges())} | Hidden Isolated Nodes: {len(isolated_nodes)}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
             )
             
    return fig