import os
import pickle
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from tqdm import tqdm

def generate_node2vec_embeddings(
    graphs, 
    dimensions=32, 
    walk_length=20, 
    num_walks=100, 
    workers=4, 
    p=1, 
    q=1
):
    """
    Generates node embeddings for a sequence of graphs.
    Returns a sequence of embedding matrices (as dictionaries or numpy arrays).
    """
    embeddings_sequence = []
    
    # Collect all unique nodes across all graphs to ensure consistent alignment
    # If the nodes are consistent across time, this is straightforward.
    all_nodes = set()
    for g in graphs:
        if g is not None:
            all_nodes.update(g.nodes())
            
    all_nodes = sorted(list(all_nodes))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    for i, g in enumerate(tqdm(graphs, desc="Processing graphs")):
        if g is None or len(g.nodes()) == 0:
            # Handle empty or None graph by padding with zeros
            emb_matrix = np.zeros((len(all_nodes), dimensions))
            embeddings_sequence.append(emb_matrix)
            continue
            
        # Fit Node2Vec
        # quiet=True reduces spam, use standard parameters
        n2v = Node2Vec(g, dimensions=dimensions, walk_length=walk_length, 
                       num_walks=num_walks, workers=workers, p=p, q=q, quiet=True)
                       
        model = n2v.fit(window=10, min_count=1, batch_words=4)
        
        # Build the embedding matrix perfectly aligned with all_nodes
        emb_matrix = np.zeros((len(all_nodes), dimensions))
        for node in g.nodes():
            if str(node) in model.wv:
                idx = node_to_idx[node]
                emb_matrix[idx] = model.wv[str(node)]
                
        embeddings_sequence.append(emb_matrix)
        
    return embeddings_sequence, node_to_idx

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    GRAPH_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "GraphAnalysis", "DynamicGraphPkls"))
    EMBEDDING_DIR = os.path.join(BASE_DIR, "GraphEmbeddings")
    
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    
    WINDOW_SIZE = 7
    STEP_SIZE = 1
    DISTANCE_METHODS = ["euclidean"]
    PERCENTILE_LIST = [0.1]
    
    DIMENSIONS = 32
    
    for method in DISTANCE_METHODS:
        print(f"\n================ Processing method: {method} ================")
        method_dir = os.path.join(GRAPH_DIR, method)
        
        if not os.path.exists(method_dir):
            print(f"Directory not found: {method_dir}")
            continue
            
        for p in PERCENTILE_LIST:
            pkl_filename = f"dynamic_graphs_{method}_Window{WINDOW_SIZE}_Step{STEP_SIZE}_p{p}.pkl"
            pkl_path = os.path.join(method_dir, pkl_filename)
            
            if not os.path.exists(pkl_path):
                print(f"File not found: {pkl_path}")
                continue
                
            out_filename = f"dynamic_embeddings_{method}_Window{WINDOW_SIZE}_Step{STEP_SIZE}_p{p}.pkl"
            out_path = os.path.join(EMBEDDING_DIR, out_filename)
            
            if os.path.exists(out_path):
                print(f"Embeddings already exist for {method} p{p}, skipping...")
                continue
                
            print(f"Loading {pkl_path}...")
            with open(pkl_path, 'rb') as f:
                graphs = pickle.load(f)
                
            print(f"Generating Node2Vec embeddings for {method} p{p}...")
            embeddings_seq, node_mapping = generate_node2vec_embeddings(
                graphs, dimensions=DIMENSIONS
            )
            
            # embeddings_seq is a sequence of np arrays, shape (num_days, num_nodes, emb_dim)
            embeddings_array = np.array(embeddings_seq)
            
            save_data = {
                'embeddings': embeddings_array,
                'node_mapping': node_mapping,
                'method': method,
                'percentile': p,
                'dimensions': DIMENSIONS
            }
            
            print(f"Saving embeddings shape {embeddings_array.shape} to {out_path}...")
            with open(out_path, 'wb') as f:
                pickle.dump(save_data, f)
                
    print("\nEmbedding generation complete!")