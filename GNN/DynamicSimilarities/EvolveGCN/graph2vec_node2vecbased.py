import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import time
import random

class Node2VecAggregator:
    def __init__(self, dimensions=64, walk_length=10, num_walks=10, window=5, min_count=1, workers=1, pooling='mean', seed=42):
        """
        Generates Graph embeddings by pooling Node2Vec node embeddings.
        Supports pooling: 'mean', 'sum', 'max'
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.pooling = pooling.lower()
        self.seed = seed
        self.embeddings = None

    def _generate_random_walks(self, graph):
        """Standard deepwalk/node2vec style unbiased random walks."""
        walks = []
        nodes = list(graph.nodes())
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = [str(node)]
                curr_node = node
                for _ in range(self.walk_length - 1):
                    neighbors = list(graph.neighbors(curr_node))
                    if not neighbors:
                        break
                    next_node = random.choice(neighbors)
                    walk.append(str(next_node))
                    curr_node = next_node
                walks.append(walk)
        return walks

    def _pool_embeddings(self, node_vectors):
        """Pools an array of node embeddings into a single graph embedding."""
        if len(node_vectors) == 0:
            return np.zeros(self.dimensions)
            
        if self.pooling == 'mean':
            return np.mean(node_vectors, axis=0)
        elif self.pooling == 'sum':
            return np.sum(node_vectors, axis=0)
        elif self.pooling == 'max':
            return np.max(node_vectors, axis=0)
        else:
            raise ValueError(f"Pooling method '{self.pooling}' not supported. Use 'mean', 'sum', or 'max'.")

    def fit(self, graphs):
        """
        graphs: List of networkx graphs
        Trains node embeddings per graph, then pools into graph embeddings.
        """
        print(f"Generating Node2Vec walks for {len(graphs)} graphs...")
        self.embeddings = []
        
        for i, G in enumerate(graphs):
            if G.number_of_nodes() == 0:
                self.embeddings.append(np.zeros(self.dimensions))
                continue
                
            # 1. Generate Walks
            walks = self._generate_random_walks(G)
            
            # 2. Train Word2Vec on the walks (Node embeddings)
            model = Word2Vec(
                sentences=walks,
                vector_size=self.dimensions,
                window=self.window,
                min_count=self.min_count,
                sg=1, # Skip-Gram (equivalent to Node2Vec's usage)
                workers=self.workers,
                seed=self.seed
            )
            
            # 3. Extract Node Vectors
            node_vectors = []
            for node in G.nodes():
                if str(node) in model.wv:
                    node_vectors.append(model.wv[str(node)])
                else:
                    node_vectors.append(np.zeros(self.dimensions))
                    
            node_vectors = np.array(node_vectors)
            
            # 4. Pool them into a Graph Vector
            graph_vector = self._pool_embeddings(node_vectors)
            self.embeddings.append(graph_vector)
            
        self.embeddings = np.array(self.embeddings)

    def get_embedding(self):
        return self.embeddings

# --- Updated Integration ---

def get_node2vec_pooled_embeddings(graphs, dimensions=20, pooling='mean', workers=1, walk_length=10, num_walks=10, seed=42):
    start_time = time.time()
    
    valid_graphs = []
    valid_indices = []
    
    for i, G in enumerate(graphs):
        if G.number_of_nodes() > 0:
            valid_graphs.append(G)
            valid_indices.append(i)
            
    print(f"Generating Node2Vec {pooling.capitalize()}-Pooled embeddings for {len(valid_graphs)} valid graphs...")
            
    # Use our Node2Vec pooling implementation
    model = Node2VecAggregator(
        dimensions=dimensions, 
        walk_length=walk_length, 
        num_walks=num_walks, 
        workers=workers, 
        pooling=pooling, 
        seed=seed
    )
    model.fit(valid_graphs)
    
    emb = model.get_embedding()
    
    graph_embeddings = np.zeros((len(graphs), dimensions))
    for idx, valid_idx in enumerate(valid_indices):
        graph_embeddings[valid_idx] = emb[idx]
        
    elapsed_time = time.time() - start_time
    print(f"Finished! Total time: {elapsed_time:.2f}s")
    
    # Return two values to match the wrapper expectations (if requested)
    return graph_embeddings, np.zeros((len(graphs), 1))