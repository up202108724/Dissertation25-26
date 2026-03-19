import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

def save_graph_plot(G: nx.Graph, strategy_name: str, output_folder: str = "graph_plots", node_categories: dict = None):
    """
    Plots and saves a NetworkX graph to a specified folder.
    The file is named dynamically using the strategy used and the number of nodes.
    
    Parameters:
    - G (nx.Graph): The networkx graph to plot.
    - strategy_name (str): Name of the metric/strategy (e.g., "StandardScaled", "CID", "DTW").
    - output_folder (str): Directory where the image will be saved.
    - node_categories (dict, optional): Mapping structure of node_id -> category name.
    """
    # 0. Filter nodes to keep only those with at least one edge
    G = G.subgraph([n for n, d in G.degree() if d > 0]).copy()

    # 1. Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # 2. Get the number of nodes
    num_nodes = G.number_of_nodes()
    
    # 3. Create the dynamic filename (e.g., standard_scaled_50_nodes.png)
    safe_strategy_name = strategy_name.lower().replace(" ", "_").replace("-", "_")
    
    start_date = G.graph.get("start_date")
    end_date = G.graph.get("end_date")
    
    if start_date is not None and end_date is not None:
        sd_str = str(start_date).replace(":", "-").replace(" ", "_").split("T")[0]
        ed_str = str(end_date).replace(":", "-").replace(" ", "_").split("T")[0]
        filename = f"{safe_strategy_name}_{sd_str}_to_{ed_str}_{num_nodes}_nodes.png"
        title_suffix = f"\nWindow: {start_date} to {end_date}"
    else:
        filename = f"{safe_strategy_name}_{num_nodes}_nodes.png"
        title_suffix = ""

    output_file_path = os.path.join(output_folder, filename)
    
    # 4. Set up the matplotlib figure
    plt.figure(figsize=(14, 10)) # Slightly wider figure to accommodate legend on the right
    pos = nx.spring_layout(G, seed=42) # Seed for reproducible layouts
    
    # 5. Draw the graph with optional categorization
    if node_categories is not None:
        # Determine unique categories and assign a distinct color to each
        unique_cats = list(set(node_categories.values()))
        cmap = plt.get_cmap('tab20')
        cat_color_map = {cat: cmap(i / max(1, len(unique_cats) - 1)) for i, cat in enumerate(unique_cats)}
        
        # Match each graph node to its designated color (fallback to 'lightgray')
        node_colors = [cat_color_map.get(node_categories.get(node), "lightgray") for node in G.nodes()]
        
        nx.draw(
            G, 
            pos, 
            with_labels=True, 
            node_color=node_colors, 
            edge_color='lightgray', 
            node_size=600, 
            font_size=8,
            font_weight='bold'
        )
        
        # Add Legend
        legend_handles = [mpatches.Patch(color=color, label=str(cat)) for cat, color in cat_color_map.items()]
        plt.legend(handles=legend_handles, title="Categories", loc="upper left", bbox_to_anchor=(1, 1))
    
    else:
        nx.draw(
            G, 
            pos, 
            with_labels=True, 
            node_color='skyblue', 
            edge_color='lightgray', 
            node_size=600, 
            font_size=8,
            font_weight='bold'
        )
    
    # Add an informative title
    plt.title(f"Product Graph: {strategy_name} Strategy ({num_nodes} nodes){title_suffix}", fontsize=14)
    
    # 6. Save the figure to the folder and close memory
    plt.savefig(output_file_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close()
    
    print(f"Graph automatically saved to: {output_file_path}")

def save_dynamic_graph_plots(graphs: list, strategy_name: str, base_output_folder: str = "dynamic_graph_plots", node_categories: dict = None):
    """
    Saves a series of dynamic graphs into a specific subfolder.
    
    Parameters:
    - graphs: A list of networkx graphs (e.g., returned from build_dynamic_similarity_graphs).
    - strategy_name: Base strategy or metric name.
    - base_output_folder: Root folder to save these specific dynamic runs.
    - node_categories: Optional mapping for coloring.
    """
    safe_strategy_name = strategy_name.lower().replace(" ", "_").replace("-", "_")
    output_folder = os.path.join(base_output_folder, safe_strategy_name)
    
    print(f"Saving {len(graphs)} dynamic graph plots into {output_folder}...")
    
    for i, G in enumerate(graphs):
        # By setting the output folder here, we route all individual graphs cleanly
        save_graph_plot(
            G=G, 
            strategy_name=strategy_name, 
            output_folder=output_folder, 
            node_categories=node_categories
        )
    
    print("All dynamic graphs have been saved.")