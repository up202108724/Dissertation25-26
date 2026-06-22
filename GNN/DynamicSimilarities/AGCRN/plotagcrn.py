import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load the learned adjacency matrix
# Shape should be (61, 61) based on your benchmark products
adj_matrix = np.load("mtgnn_learned_adjacency.npy")
num_nodes = adj_matrix.shape[0]

# 2. Define the visual threshold
# Edges with a weight below this value will NOT be drawn.
# Adjust this between 0.01 and 0.10 to find the cleanest visual layout.
threshold = 0.05 

# 3. Initialize a Directed Graph
# We use DiGraph because MTGNN learns asymmetric relationships (A -> B is not B -> A)
G = nx.DiGraph()
G.add_nodes_from(range(num_nodes))

# 4. Populate the graph with edges above the threshold
for i in range(num_nodes):
    for j in range(num_nodes):
        weight = adj_matrix[i, j]
        # Skip self-loops and weak edges for a cleaner plot
        if i != j and weight > threshold:
            # Note: In MTGNN, adj[i, j] typically implies information flows from j to i.
            # We map this as an edge from j (source) to i (target).
            G.add_edge(j, i, weight=weight)

# 5. Graph Layout and Aesthetics
plt.figure(figsize=(14, 12))

# spring_layout uses a force-directed algorithm to cluster highly connected nodes together
pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42) 

# Extract edge weights to dynamically scale the thickness of the lines
edges = G.edges()
weights = [G[u][v]['weight'] * 5 for u, v in edges] # Multiplier controls line thickness

# 6. Draw the Network
# Draw Nodes
nx.draw_networkx_nodes(
    G, pos, 
    node_size=400, 
    node_color='lightblue', 
    edgecolors='black',
    linewidths=1.5
)

# Draw Edges (curved slightly to handle bidirectional connections gracefully)
nx.draw_networkx_edges(
    G, pos, 
    edgelist=edges, 
    width=weights, 
    arrowstyle='-|>', 
    arrowsize=15, 
    edge_color='gray', 
    alpha=0.7,
    connectionstyle='arc3,rad=0.1' 
)

# Draw Node Labels (Product IDs/Indices)
nx.draw_networkx_labels(
    G, pos, 
    font_size=9, 
    font_family='sans-serif', 
    font_weight='bold'
)

# 7. Final Polish
plt.title(f"MTGNN Learned Adaptive Topology (Edge Weight > {threshold})", fontsize=18, pad=20)
plt.axis('off') # Hide the grid axes
plt.tight_layout()

# Save for your dissertation
plt.savefig("mtgnn_networkx_topology.png", dpi=300, bbox_inches='tight')
plt.show()

# 8. Print Topological Diagnostics
print("=== MTGNN Graph Diagnostics ===")
print(f"Total Products (Nodes): {num_nodes}")
print(f"Total Edges Plotted: {G.number_of_edges()}")
print(f"Average In-Degree (Sources per Target): {G.number_of_edges() / num_nodes:.2f}")

# Find the biggest "Hero Product" (Node with the highest out-degree)
out_degrees = dict(G.out_degree())
hero_node = max(out_degrees, key=out_degrees.get)
print(f"Most Influential Node (Index {hero_node}) influences {out_degrees[hero_node]} other products.")