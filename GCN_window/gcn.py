import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch

class GCN_LSTM_WeightedEdges(nn.Module):
    def __init__(self, in_features, seq_len, gcn_hidden_dim, lstm_hidden_dim, out_features, lstm_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.gcn1 = pyg_nn.GCNConv(in_features, gcn_hidden_dim)
        self.gcn2 = pyg_nn.GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.bn1 = nn.BatchNorm1d(gcn_hidden_dim)
        self.bn2 = nn.BatchNorm1d(gcn_hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(
            input_size=gcn_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(lstm_hidden_dim, out_features)

    def forward(self, x, edge_index, edge_weight, batch=None):
        """
        x: [batch_size * num_nodes, seq_len, in_features] for batched input
        edge_index: [2, num_edges]
        edge_weight: [num_edges]
        batch: [batch_size * num_nodes] (batch assignment for nodes)
        """
        # If batched, x is [batch_size * num_nodes, seq_len, in_features]
        # If not batched, x is [num_nodes, seq_len, in_features]
        # print(x.shape)
        # Process each time step with GCN
        gcn_outputs = []
        for t in range(self.seq_len):
            x_t = x[:, t, :]  # [batch_size * num_nodes, in_features]
            gcn_out = self.gcn1(x_t, edge_index, edge_weight=edge_weight)
            gcn_out = self.bn1(gcn_out)
            gcn_out = torch.relu(gcn_out)
            gcn_out = self.dropout(gcn_out)
            gcn_out = self.gcn2(gcn_out, edge_index, edge_weight=edge_weight)
            gcn_out = self.bn2(gcn_out)
            gcn_out = torch.relu(gcn_out)
            gcn_out = self.dropout(gcn_out)
            gcn_outputs.append(gcn_out)

        # Stack GCN outputs: [batch_size * num_nodes, seq_len, gcn_hidden_dim]
        gcn_out = torch.stack(gcn_outputs, dim=1)

        # Reshape for LSTM: [batch_size, num_nodes, seq_len, gcn_hidden_dim]
        lstm_out, _ = self.lstm(gcn_out)  # [num_nodes, seq_len, lstm_hidden_dim]
        lstm_out = lstm_out[:, -1, :]  # [num_nodes, lstm_hidden_dim]

        out = self.fc(lstm_out)  # [batch_size * num_nodes, out_features]
        return out