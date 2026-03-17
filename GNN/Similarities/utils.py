

import torch
from torch_geometric.data import Data

def nx_to_pyg_data(G, df_scaled, y=None):

    node_ids = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_ids)}
    x = torch.tensor(df_scaled[node_ids].T.values, dtype=torch.float)

    print("Node features shape:", x.shape)
    # Edge index
    edges = list(G.edges(data=True))
    if len(edges) > 0:
        edge_index_list = []
        edge_attr_list = []

        for u, v, attr in edges:
            i, j = node_to_idx[u], node_to_idx[v]
            w = attr.get("weight", 1.0)

            # Undirected graph -> store both directions
            edge_index_list.append([i, j])
            edge_index_list.append([j, i])

            edge_attr_list.append([w])
            edge_attr_list.append([w])

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).T.contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float)

    data.node_ids = node_ids

    return data