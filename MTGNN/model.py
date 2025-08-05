import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse


class GraphConstructor(nn.Module):
    def __init__(self, num_nodes, emb_dim, device):
        super().__init__()
        self.emb1 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.emb2 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.device = device

    def forward(self):
        # Compute similarity
        adj = F.relu(torch.mm(self.emb1, self.emb2.t()))
        adj = adj - torch.diag_embed(torch.diagonal(adj))  # Remove self-loops
        adj = F.softmax(adj, dim=-1)
        return adj


class MixProp(nn.Module):
    def __init__(self, in_dim, out_dim, num_hops=2, dropout=0.3):
        super().__init__()
        self.mlp = nn.Linear((num_hops + 1) * in_dim, out_dim)
        self.num_hops = num_hops
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        out = [x]
        x_next = x
        for _ in range(self.num_hops):
            x_next = torch.matmul(adj, x_next)
            out.append(x_next)
        x_cat = torch.cat(out, dim=-1)
        return self.dropout(F.relu(self.mlp(x_cat)))


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, kernel_size - 1))

    def forward(self, x):
        # x: [B, N, F, T]
        x = x.permute(0, 2, 1, 3)  # [B, F, N, T]
        out = self.conv(x)[:, :, :, :-2]  # remove extra padding
        return out.permute(0, 2, 1, 3)  # [B, N, F, T]


class MTGNNBlock(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, gcn_hops, device):
        super().__init__()
        self.temporal = TemporalConv(in_dim, out_dim)
        self.graph_constructor = GraphConstructor(num_nodes, emb_dim=10, device=device)
        self.mixprop = MixProp(out_dim, out_dim, num_hops=gcn_hops)

    def forward(self, x):
        # x: [B, N, F, T]
        x_t = self.temporal(x)  # [B, N, F, T]
        B, N, F, T = x_t.shape
        x_last = x_t[:, :, :, -1]  # Use last timestep for GCN: [B, N, F]
        adj = self.graph_constructor()  # [N, N]
        x_g = self.mixprop(x_last, adj)  # [B, N, F]
        return x_g.unsqueeze(-1)  # [B, N, F, 1]


class MTGNN(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim, gcn_hops=2, device='cpu'):
        super().__init__()
        self.block1 = MTGNNBlock(num_nodes, in_dim, hidden_dim, gcn_hops, device)
        self.block2 = MTGNNBlock(num_nodes, hidden_dim, hidden_dim, gcn_hops, device)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: [B, N, F, T]
        x = self.block1(x)
        x = self.block2(x)
        x = x.squeeze(-1)  # [B, N, F]
        return self.output_proj(x)  # [B, N, out_dim]


# === Example Usage ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTGNN(num_nodes=12, in_dim=1, hidden_dim=32, out_dim=1, gcn_hops=2, device=device).to(device)

    batch_size = 32
    num_nodes = 12
    in_dim = 1
    seq_len = 12

    x = torch.randn(batch_size, num_nodes, in_dim, seq_len).to(device)
    out = model(x)
    print("Output shape:", out.shape)  # Expected: [B, N, 1]
