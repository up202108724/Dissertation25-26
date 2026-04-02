import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def generate_and_save_graph_plots(graphs, sim_dfs, plot_dir, method_name, metric_type="Distance"):
    """
    Consolidates the graph analysis plotting logic to create and save all visual graphics
    related to the dynamic temporal graphs.
    
    Args:
        graphs (list): A list of networkx.Graph objects spanning across temporal windows.
        sim_dfs (list): A list of pandas DataFrames or numpy arrays representing the distance/similarity matrices per window.
        plot_dir (str): The folder path where plots will be saved.
        method_name (str): The name of the distance/similarity method (e.g., 'CID', 'AMPLITUDE_OFFSET').
        metric_type (str): Usually 'Distance' or 'Similarity' to correctly label the axes.
    """
    
    method_name = str(method_name).upper()
    os.makedirs(plot_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Number of Edges Evolution per Window (Line Plot)
    # -------------------------------------------------------------------------
    edges_per_window = [g.number_of_edges() for g in graphs]
    
    plt.figure(figsize=(15, 6))
    plt.plot(range(len(edges_per_window)), edges_per_window, marker='o', linestyle='-', color='b', markersize=4)
    plt.title(f"Number of Edges per Temporal Window ({method_name})")
    plt.xlabel("Window Index")
    plt.ylabel("Number of Edges")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'edges_per_window.png'), dpi=300, bbox_inches='tight')
    plt.close() # Close to free up memory

    # -------------------------------------------------------------------------
    # 2. Overall Distribution of Edges (Histogram / KDE)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    sns.histplot(edges_per_window, bins=30, kde=True, color='skyblue')
    plt.title(f"Distribution of the Number of Edges Across All Windows ({method_name})")
    plt.xlabel("Number of Edges")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'edges_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Value Distributions Across All Time Windows (Boxplot)
    # -------------------------------------------------------------------------
    all_values_by_window = []
    
    for w_idx, adj_matrix in enumerate(sim_dfs):
        # Handle depending on input (NumPy arrays or Pandas DataFrame)
        if isinstance(adj_matrix, pd.DataFrame):
            vals = adj_matrix.values
        else:
            vals = adj_matrix
            
        # Get lower triangle indices (excluding diagonal) to avoid duplicate edges & self-loops
        mask = np.tril(np.ones(vals.shape), k=-1).astype(bool)
        lower_tri_vals = vals[mask]
        
        df_temp = pd.DataFrame({
            'Window': [w_idx] * len(lower_tri_vals),
            'Value': lower_tri_vals
        })
        all_values_by_window.append(df_temp)

    if len(all_values_by_window) > 0:
        df_plot = pd.concat(all_values_by_window, ignore_index=True)
        
        plt.figure(figsize=(18, 7))
        sns.boxplot(data=df_plot, x='Window', y='Value', color='skyblue', fliersize=1)

        plt.title(f"Distribution of {method_name} {metric_type}s Across Time Windows")
        plt.xlabel("Window Index")
        plt.ylabel(f"{method_name} Value")

        # Rotate x-axis labels if there are too many windows
        num_windows = len(sim_dfs)
        if num_windows > 30:
            plt.xticks(ticks=range(0, num_windows, 5), labels=range(0, num_windows, 5), rotation=45)
        else:
            plt.xticks(rotation=45)
            
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'value_distribution_all_windows.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # -------------------------------------------------------------------------
    # 4. Boxplot for the window with the MAXIMUM number of edges
    # -------------------------------------------------------------------------
    if len(edges_per_window) > 0:
        max_edges_window_idx = np.argmax(edges_per_window)
        max_edges_count = edges_per_window[max_edges_window_idx]
        
        adj_matrix_max = sim_dfs[max_edges_window_idx]
        if isinstance(adj_matrix_max, pd.DataFrame):
            vals_max = adj_matrix_max.values
        else:
            vals_max = adj_matrix_max

        mask_max = np.tril(np.ones(vals_max.shape), k=-1).astype(bool)
        lower_tri_vals_max = vals_max[mask_max]

        plt.figure(figsize=(8, 6))
        sns.boxplot(y=lower_tri_vals_max, color='lightgreen')

        plt.title(f"Distribution of {method_name} {metric_type}s for Window {max_edges_window_idx}\n(Highest Edge Count: {max_edges_count})")
        plt.ylabel(f"{method_name} Value")

        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'distribution_window_{max_edges_window_idx}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"✅ Successfully saved {len(graphs)} graphical plots to '{os.path.abspath(plot_dir)}'.")

def generate_and_save_node_stats(graphs, output_csv_path):
    """
    Calculates degree, betweenness, and eigenvector centrality for nodes across all temporal graphs,
    computes stability statistics, and saves them to a CSV file.
    
    Args:
        graphs (list): A list of networkx.Graph objects spanning across temporal windows.
        output_csv_path (str): The full path where the resulting CSV should be saved.
        
    Returns:
        pd.DataFrame: The dataframe containing all calculated node statistics.
    """
    node_degree_history = {}
    node_betweenness_history = {}
    node_eigenvector_history = {}

    print("Calculating degree and centrality metrics across all windows. This may take a moment...")
    for w_idx, g in enumerate(graphs):
        # Weight handling: for betweenness we ignore weights conceptually.
        # For eigenvector, weight represents connection strength.
        betweenness = nx.betweenness_centrality(g, weight=None)
        
        try:
            eigenvector = nx.eigenvector_centrality(g, max_iter=500, weight='weight')
        except nx.PowerIterationFailedConvergence:
            eigenvector = {n: 0 for n in g.nodes()} # Fallback if it fails

        for node in g.nodes():
            if node not in node_degree_history:
                node_degree_history[node] = []
                node_betweenness_history[node] = []
                node_eigenvector_history[node] = []
                
            node_degree_history[node].append(g.degree[node])
            node_betweenness_history[node].append(betweenness.get(node, 0))
            node_eigenvector_history[node].append(eigenvector.get(node, 0))

    # Compute statistics across time
    node_stats = []
    for node in node_degree_history.keys():
        degrees = node_degree_history[node]
        betweenness_vals = node_betweenness_history[node]
        eigenvector_vals = node_eigenvector_history[node]
        
        node_stats.append({
            'Node': node,
            'Mean_Degree': np.mean(degrees),
            'Median_Degree': np.median(degrees),
            'Std_Degree': np.std(degrees),
            'Max_Degree': np.max(degrees),
            'Min_Degree': np.min(degrees),
            'Zero_Degree_Windows': sum(1 for d in degrees if d == 0),
            'Mean_Betweenness': np.mean(betweenness_vals),
            'Mean_Eigenvector': np.mean(eigenvector_vals)
        })

    df_node_stats = pd.DataFrame(node_stats)

    # Calculate combined Stability Score
    # Add a small epsilon to prevent division by zero
    df_node_stats['Degree_Stability'] = df_node_stats['Mean_Degree'] / (df_node_stats['Std_Degree'] + 1e-5)

    # Sort by Eigenvector Centrality by default
    df_node_stats = df_node_stats.sort_values(by='Mean_Eigenvector', ascending=False).reset_index(drop=True)
    
    # Save the dataframe map to disk
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_node_stats.to_csv(output_csv_path, index=False)
    
    print(f"✅ Successfully calculated node stats and saved to '{os.path.abspath(output_csv_path)}'.")
    return df_node_stats

