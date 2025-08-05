import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout_p=0.3):
        super(GraphConv, self).__init__()
        self.conv = GCNConv(in_features, out_features)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, edge_index, edge_weight=None):
        # x: (batch_size * num_nodes, in_features, seq_len)
        batch_nodes, in_features, seq_len = x.size()
        output = torch.zeros(batch_nodes, self.conv.out_channels, seq_len, device=x.device)
        for t in range(seq_len):
            out_t = self.conv(x[:, :, t], edge_index, edge_weight)
            out_t = self.dropout(out_t)
            output[:, :, t] = out_t
        return output

class GatedTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_p=0.3):
        super(GatedTemporalConv, self).__init__()
        # Output 2 * out_channels for P and Q in GLU
        self.conv = nn.Conv1d(in_channels, 2 * out_channels, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # x: (batch_nodes, channels, seq_len)
        out = self.conv(x)  # (batch_nodes, 2 * out_channels, seq_len)
        P, Q = torch.split(out, out.size(1)//2, dim=1)
        gated = P * torch.sigmoid(Q)  # (batch_nodes, out_channels, seq_len)
        return self.dropout(gated)

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, out_features, seq_len,
                 kernel_size=3, dropout_p=0.3):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.kernel_size = kernel_size

        # Spatial-temporal blocks: Temporal -> Graph -> ReLU -> Temporal
        self.st_block1 = nn.Sequential(
            GatedTemporalConv(in_features, hidden_dim, kernel_size, dropout_p),
            GraphConv(hidden_dim, hidden_dim, dropout_p),
            nn.ReLU(),
            GatedTemporalConv(hidden_dim, hidden_dim, kernel_size, dropout_p)
        )

        self.st_block2 = nn.Sequential(
            GatedTemporalConv(hidden_dim, hidden_dim, kernel_size, dropout_p),
            GraphConv(hidden_dim, hidden_dim, dropout_p),
            nn.ReLU(),
            GatedTemporalConv(hidden_dim, hidden_dim, kernel_size, dropout_p)
        )

        # Final temporal convolution to reduce sequence length
        self.final_temp_conv = nn.Conv1d(hidden_dim, out_features, seq_len)

    def forward(self, x, edge_index, edge_weight=None):
        # x: (batch_size * num_nodes, in_features, seq_len)
        # ST block 1
        x = self.st_block1[0](x)
        x = self.st_block1[1](x, edge_index, edge_weight)
        x = self.st_block1[2](x)
        x = self.st_block1[3](x)

        # ST block 2
        x = self.st_block2[0](x)
        x = self.st_block2[1](x, edge_index, edge_weight)
        x = self.st_block2[2](x)
        x = self.st_block2[3](x)

        # Reduce sequence length to 1
        x = self.final_temp_conv(x)
        x = x.squeeze(-1)  # (batch_nodes, out_features)
        batch_size = x.size(0) // self.num_nodes
        x = x.view(batch_size, self.num_nodes, -1)
        return x
