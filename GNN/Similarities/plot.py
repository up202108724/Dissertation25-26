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
    # 1. Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # 2. Get the number of nodes
    num_nodes = G.number_of_nodes()
    
    # 3. Create the dynamic filename (e.g., standard_scaled_50_nodes.png)
    safe_strategy_name = strategy_name.lower().replace(" ", "_").replace("-", "_")
    filename = f"{safe_strategy_name}_{num_nodes}_nodes.png"
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
    plt.title(f"Product Graph: {strategy_name} Strategy ({num_nodes} nodes)", fontsize=14)
    
    # 6. Save the figure to the folder and close memory
    plt.savefig(output_file_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close()
    
    print(f"Graph automatically saved to: {output_file_path}")