import numpy as np
import networkx as nx
from sklearn.covariance import LedoitWolf, GraphicalLasso
from sklearn.preprocessing import StandardScaler

def apply_ifn(G, metric_type="similarity"):
    '''
    Applies an Information Filtering Network (Minimum/Maximum Spanning Tree).
    Removes cycles but retains the backbone of the graph.
    
    Args:
        G (nx.Graph): The input Ego-Graph.
        metric_type (str): "similarity" (e.g. Pearson/Spearman) or "distance" (e.g. DTW/Euclidean).
    Returns:
        nx.Graph: Filtered graph (MST).
    '''
    if G.number_of_nodes() <= 1:
        return G.copy()
        
    if metric_type == "distance":
        # For distances, we want the paths with the MINIMUM total distance
        filtered_G = nx.minimum_spanning_tree(G, weight="weight")
    else:
        # For similarities, we want the paths with the MAXIMUM total similarity
        filtered_G = nx.maximum_spanning_tree(G, weight="weight")
        
    return filtered_G

def apply_covariance_shrinkage(node_ts_data):
    '''
    Applies Covariance Shrinkage (Ledoit-Wolf) to stabilize the covariance matrix 
    when the temporal window (T) is comparable to the number of nodes (N).
    
    Args:
        node_ts_data (np.ndarray): Time series data of the nodes (shape: [T time_steps, N nodes]).
    Returns:
        np.ndarray: The shrunk empirical covariance matrix (shape: [N, N]).
    '''
    if len(node_ts_data) == 0:
        return np.array([])
        
    # StandardScaler helps stabilize the covariance calculation
    X = StandardScaler().fit_transform(node_ts_data)
    
    # Ledoit-Wolf automatically computes the optimal shrinkage intensity
    lw = LedoitWolf()
    shrunk_cov = lw.fit(X).covariance_
    
    return shrunk_cov

def apply_graphical_lasso(node_ts_data, node_ids, alpha=0.05):
    '''
    Applies Graphical Lasso to estimate a sparse precision matrix (inverse covariance).
    Filters out indirect correlations and returns a sparse Ego-Graph with direct conditional dependencies.
    
    Args:
        node_ts_data (np.ndarray): Time series data of the nodes (shape: [T time_steps, N nodes]).
        node_ids (list): Identifiers mapping to the columns (nodes) in node_ts_data.
        alpha (float): L1 regularization parameter. Higher = fewer edges (more sparse).
    Returns:
        nx.Graph: Ego-Graph containing only direct partial correlations.
        np.ndarray: The Estimated Precision Matrix.
    '''
    if len(node_ts_data) == 0 or len(node_ids) <= 1:
        G_filtered = nx.Graph()
        G_filtered.add_nodes_from(node_ids)
        return G_filtered, None

    # Graphical Lasso expects centered/standardized data
    X = StandardScaler().fit_transform(node_ts_data)
    
    # Fit GLasso
    glasso = GraphicalLasso(alpha=alpha, max_iter=500)
    try:
        glasso.fit(X)
        precision_matrix = glasso.precision_
    except Exception as e:
        print(f"Graphical Lasso failed to converge: {e}")
        G_filtered = nx.Graph()
        G_filtered.add_nodes_from(node_ids)
        return G_filtered, None
    
    G_filtered = nx.Graph()
    G_filtered.add_nodes_from(node_ids)
    
    # Build graph from precision matrix (off-diagonal elements)
    n_nodes = len(node_ids)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            val = precision_matrix[i, j]
            # If the partial correlation is non-zero, there is a direct edge
            # (thresholding against numerical artifacts)
            if abs(val) > 1e-5:
                # Store the absolute strength of the partial correlation as the weight
                G_filtered.add_edge(node_ids[i], node_ids[j], weight=abs(val))
                
    return G_filtered, precision_matrix
