import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, summary


class TemporalGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TemporalGCN, self).__init__()
        heads = 4
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.3)  # Add dropout for regularization
        # self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        # self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.3)
        # self.gat3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)  # Final layer: no head concat
        # self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # First GCN Layer
        x = self.conv1(x, edge_index, edge_weight)
        # print(x)
        # x = self.dropout(x)  # Dropout for better generalization
        x = F.leaky_relu(x)

        # Second GCN Layer
        x = self.conv2(x, edge_index, edge_weight)
        x = F.leaky_relu(x)
        # x = self.dropout(x)

        # Third GCN Layer (Final Output)
        x = self.conv3(x, edge_index, edge_weight)
        # x = F.leaky_relu(x)


        return x



