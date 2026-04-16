import numpy as np
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import hashlib
import time

class CustomGraph2Vec:
    def __init__(self, dimensions=64, wl_iterations=2, epochs=10, learning_rate=0.025, workers=4):
        self.dimensions = dimensions
        self.wl_iterations = wl_iterations
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.workers = workers
        self.model = None
        self.embeddings = None

    def _get_wl_subgraphs(self, graph):
        """
        Generates Weisfeiler-Lehman features for a single graph.
        """
        # Initialize labels: use degree if no labels exist
        labels = {node: str(graph.degree(node)) for node in graph.nodes()}
        subgraphs = list(labels.values())

        for _ in range(self.wl_iterations):
            new_labels = {}
            for node in graph.nodes():
                # Get labels of neighbors
                neighbors = sorted([labels[neigh] for neigh in graph.neighbors(node)])
                # Create a composite string: "current_label" + "sorted_neighbor_labels"
                composite = labels[node] + "".join(neighbors)
                # Hash for brevity/consistency
                new_labels[node] = hashlib.md5(composite.encode()).hexdigest()
            
            labels = new_labels
            subgraphs.extend(list(labels.values()))
        
        return subgraphs

    def fit(self, graphs):
        """
        graphs: List of networkx graphs
        """
        print(f"Extracting WL features for {len(graphs)} graphs...")
        tagged_data = []
        
        for i, G in enumerate(graphs):
            words = self._get_wl_subgraphs(G)
            # TaggedDocument takes (list of words, list of tags)
            tagged_data.append(TaggedDocument(words=words, tags=[str(i)]))

        print("Training Doc2Vec model...")
        self.model = Doc2Vec(
            vector_size=self.dimensions,
            window=0,  # Since order of subgraphs doesn't matter, we use a large context or DBOW
            min_count=1,
            dm=0,      # Distributed Bag of Words (PV-DBOW) - usually better for Graph2Vec
            sample=0,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate
        )
        
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        
        # Extract embeddings in the order of input graphs
        self.embeddings = np.array([self.model.dv[str(i)] for i in range(len(graphs))])

    def get_embedding(self):
        return self.embeddings

# --- Updated Integration for your code ---

def get_graph2vec_embeddings(graphs, dimensions=64, workers=4):
    start_time = time.time()
    
    valid_graphs = []
    valid_indices = []
    
    for i, G in enumerate(graphs):
        if G.number_of_nodes() > 0:
            valid_graphs.append(G)
            valid_indices.append(i)
            
    print(f"Generating Graph2Vec embeddings for {len(valid_graphs)} valid graphs...")
            
    # Use our custom implementation
    model = CustomGraph2Vec(dimensions=dimensions, workers=workers, wl_iterations=2)
    model.fit(valid_graphs)
    
    emb = model.get_embedding()
    
    graph_embeddings = np.zeros((len(graphs), dimensions))
    for idx, valid_idx in enumerate(valid_indices):
        graph_embeddings[valid_idx] = emb[idx]
        
    elapsed_time = time.time() - start_time
    print(f"Finished! Total time: {elapsed_time:.2f}s")
    
    # Return two values to match your load_or_generate_embeddings expectation
    # Target node embeddings are returned as zeros
    return graph_embeddings, np.zeros((len(graphs), 1))