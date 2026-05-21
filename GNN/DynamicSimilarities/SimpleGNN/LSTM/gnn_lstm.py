from torch_geometric.nn import GCNConv
from torch.nn import nn
class SimpleGNNLSTMForecaster(nn.Module):
    """
    GraphSAGE encoder + LSTM temporal model.

    Per training sample the pipeline is:
      L ego-graphs  →  SAGEConv ×2  →  L central-node embeddings
                    →  LSTM over the L-step sequence
                    →  linear head  →  (B, horizon, 1)

    All B×L graphs are batched together in a single PyG Batch so that the
    two SAGEConv layers run in one efficient pass.

    Node feature layout  (feature_dim = graph_window_size + cal_dim + 8):
      Target node    : [ts_window | cal_at_t | stats_8]
                       computed by compute_target_node_features_seq
      Neighbor nodes : [zeros (cal slots) right-filled with ts | stats_8]
                       computed by compute_neighbor_node_features_pure
    """

    def __init__(
        self,
        in_channels: int,        # graph_window_size + cal_dim + 8
        hidden_channels: int,
        gnn_out_channels: int,
        lstm_hidden: int,
        num_lstm_layers: int = 1,
        horizon: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, gnn_out_channels, add_self_loops=True)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(
            gnn_out_channels,
            lstm_hidden,
            num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.head    = nn.Linear(lstm_hidden, horizon)
        self.horizon = horizon

    def forward(self, pyg_batch, target_node_indices, B: int, L: int):
        """
        pyg_batch           : PyG Batch of B×L graphs in sample-major order
                              (sample_0_t0, sample_0_t1, ..., sample_1_t0, ...)
        target_node_indices : (B×L,) global indices of the central node in each
                              sub-graph; since node 0 is always central,
                              use pyg_batch.ptr[:-1].
        B                   : mini-batch size
        L                   : lookback length (LSTM sequence length)

        Returns : (B, horizon, 1)
        """
        ew = (pyg_batch.edge_attr.squeeze(1)
              if (pyg_batch.edge_attr is not None and pyg_batch.edge_attr.shape[0] > 0)
              else None)
        h = self.conv1(pyg_batch.x, pyg_batch.edge_index, edge_weight=ew)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h, pyg_batch.edge_index, edge_weight=ew)

        # Central-node embeddings: (B×L, gnn_out_channels)
        z = h[target_node_indices]

        # Reshape to temporal sequence: (B, L, sage_out_channels)
        z_seq = z.view(B, L, -1)

        # LSTM: returns (B, L, lstm_hidden)
        lstm_out, _ = self.lstm(z_seq)

        last = self.drop(lstm_out[:, -1, :])   # (B, lstm_hidden)
        out  = self.head(last)                  # (B, horizon)
        return out.unsqueeze(-1)                # (B, horizon, 1)