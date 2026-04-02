import sys
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

def build_neighborhood_graph_from_matrix(
    dist_df: pd.DataFrame,
    target_product,
    distance_method: str = "euclidean",
    distance_value_threshold: float = 1.0,
    self_loop: bool = False,
):
    """
    Builds a graph containing the target product and its 1st-order neighborhood.
    It includes the links between the neighbors if they also pass the threshold.
    """
    if target_product not in dist_df.columns:
        return None
        
    # 1. Find the 1st-order neighborhood (nodes within threshold distance from target)
    target_distances = dist_df.loc[target_product]
    valid_neighbors = target_distances[target_distances <= distance_value_threshold].index.tolist()
    
    if target_product not in valid_neighbors:
        valid_neighbors.append(target_product)
        
    # If the target product is completely isolated
    if len(valid_neighbors) == 1 and not self_loop:
        G = nx.Graph(name=f"{distance_method.capitalize()}_Neighborhood_{target_product}")
        G.add_node(target_product)
        return G
        
    # 2. Extract the sub-distance matrix for only these nodes
    sub_dist_df = dist_df.loc[valid_neighbors, valid_neighbors]
    sub_dist_matrix = sub_dist_df.values
    n = len(valid_neighbors)
    
    G = nx.Graph(name=f"{distance_method.capitalize()}_Neighborhood_{target_product}")
    G.add_nodes_from(valid_neighbors)
    
    # 3. Add edges between any nodes in this neighborhood that pass the threshold
    for i in range(n):
        for j in range(i + 1, n):
            dist_val = sub_dist_matrix[i, j]
            if dist_val <= distance_value_threshold:
                G.add_edge(valid_neighbors[i], valid_neighbors[j], weight=float(dist_val))
                
    if self_loop:
        for node in valid_neighbors:
            G.add_edge(node, node, weight=0.0)
            
    return G

def build_graphs_from_dynamic_distances(
    dist_dfs: list,
    window_info: list,
    target_product,
    distance_method: str = "dtw",
    distance_threshold: float = 1.0,
    **kwargs
):
    """
    Builds a sequence of dynamic distance graphs selectively on the target neighborhood.
    """
    graphs = []
    mapped_method = "euclidean" if distance_method == "standardscaled" else distance_method

    for dist_df, info in zip(dist_dfs, window_info):
        if distance_threshold is not None:
            G = build_neighborhood_graph_from_matrix(
                dist_df=dist_df,
                target_product=target_product,
                distance_method=mapped_method,
                distance_value_threshold=distance_threshold,
                self_loop=kwargs.get('self_loop', False)
            )
            if G is not None:
                G.graph["start_date"] = info["start_date"]
                G.graph["end_date"] = info["end_date"]
            graphs.append(G)
        else:
            graphs.append(None)
            
    return graphs

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    WINDOW_SIZE = 7
    STEP_SIZE = 1
    DISTANCE_METHODS = ["euclidean", "cid", "hamming", "amplitude_offset", "slope_consistency"]
    PERCENTILE_LIST = [0.1]
    TARGET_PRODUCT = 907969
    
    for method in DISTANCE_METHODS:
        print(f"\n================ Processing method: {method} ================")
        pkl_dir = os.path.join(BASE_DIR, "DynamicGraphPkls", method)
        pkl_path = os.path.join(pkl_dir, f"dynamic_distances_{method}_Window{WINDOW_SIZE}_Step{STEP_SIZE}.pkl")
        
        if not os.path.exists(pkl_path):
            print(f"File not found: {pkl_path}")
            continue
            
        print(f"Loading {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            distance_data = pickle.load(f)
            
        dist_dfs = distance_data['dist_dfs']
        window_info = distance_data['window_info']
        
        # Compute threshold based ONLY on the target product's distance distribution to get the relevant neighborhood
        all_dist_values = []
        for dist_df in dist_dfs:
            if TARGET_PRODUCT in dist_df.columns:
                vals = dist_df.loc[TARGET_PRODUCT].values
                vals = vals[~pd.isna(vals)].astype(np.float32)
                vals = vals[vals > 0] # exclude self
                all_dist_values.extend(vals)
                
        if not all_dist_values:
            print(f"No valid distances found for product {TARGET_PRODUCT}.")
            continue
            
        for p in PERCENTILE_LIST:
            threshold = np.percentile(all_dist_values, p * 100)
            print(f"Building neighborhood graphs for {method} with threshold {threshold:.4f} ({int(p*100)}th p)...")
            
            graphs = build_graphs_from_dynamic_distances(
                dist_dfs=dist_dfs,
                window_info=window_info,
                target_product=TARGET_PRODUCT,
                distance_method=method,
                distance_threshold=threshold
            )
            
            output_pkl = os.path.join(pkl_dir, f"dynamic_graphs_{TARGET_PRODUCT}_{method}_Window{WINDOW_SIZE}_Step{STEP_SIZE}_p{int(p*100)}.pkl")
            print(f"Saving {len([g for g in graphs if g is not None])} neighborhood graphs to {output_pkl}...")
            with open(output_pkl, 'wb') as f:
                pickle.dump(graphs, f)
                
    print("\nGraph construction complete!")