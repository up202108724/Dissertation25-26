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
        marker=dict(size=0.1, color='rgba(0,0,0,0)'), # Invisible marker
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
    
    for i, (cat, d) in enumerate(categories.items()):
        color = palette[i % len(palette)]
        node_trace = go.Scatter(
            x=d['x'], y=d['y'],
            mode='markers+text',
            hoverinfo='text',
            text=d['text'],
            hovertext=d['hovertext'],
            textposition="middle center",
            textfont=dict(size=8, color='white'),
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


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    
    # Path to one of the dynamic graph sequence pkl files (update parameters as needed)
    METHOD = "euclidean"
    WINDOW = 7
    STEP = 1
    P = 0.1
    TARGET_PRODUCT = 907969
    pkl_filename = f"dynamic_graphs_{METHOD}_Window{WINDOW}_Step{STEP}_p{P}.pkl"
    if TARGET_PRODUCT is not None:
        pkl_filename = f"{TARGET_PRODUCT}/dynamic_graphs_{METHOD}_Window{WINDOW}_Step{STEP}_p{P}.pkl"
        
    pkl_path = os.path.join(BASE_DIR, "DynamicGraphPkls", METHOD, pkl_filename)
    
    if not os.path.exists(pkl_path):
        print(f"Graph file not found: {pkl_path}")
        print("Please verify the GRAPH_WINDOW_SIZE, STEP_SIZE, and distance method variables.")
    else:
        print(f"Loading sequence from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)
            
        print(f"Loaded {len(graphs)} graphs over time.")
        
        # Create output directory
        target_dir = os.path.join(BASE_DIR, "Graphplots", METHOD, f"p{P}")
        os.makedirs(target_dir, exist_ok=True)
        print(f"Saving plots to {target_dir}...")
        
        # Iterate and save all valid graphs in the sequence
        for day_idx, G in enumerate(graphs):
             if G is not None and len(G.nodes()) > 0:
                start_date = G.graph.get('start_date', f'Day_{day_idx}')
                end_date = G.graph.get('end_date', 'Unknown')
                
                # Clean dates for safe filename usage (e.g. replacing colons or spaces)
                safe_start_date = str(start_date).replace(':', '-').replace(' ', '_')
                
                print(f"Plotting graph from {start_date} to {end_date} ({len(G.nodes())} nodes, {len(G.edges())} edges)...")
                
                save_filepath = os.path.join(target_dir, f"graph_{safe_start_date}.html")
                plot_networkx_plotly(
                    G, 
                    title=f"Graph from {start_date} to {end_date} (Method: {METHOD})", 
                    save_path=save_filepath,
                    target_node=TARGET_PRODUCT
                )
        print("Finished saving all plots!")