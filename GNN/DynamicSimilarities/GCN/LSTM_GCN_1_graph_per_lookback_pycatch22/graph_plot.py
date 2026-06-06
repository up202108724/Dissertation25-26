import os
import pickle
import networkx as nx
import plotly.graph_objects as go
import plotly.colors as pcolors

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

