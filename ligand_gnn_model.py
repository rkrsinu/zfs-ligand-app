# ==========================================================
# ligand_gnn_model.py
# EXACT model matching checkpoint keys
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


# ----------------------------------------------------------
# Custom relation-aware convolution
# ----------------------------------------------------------
class RelConv(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="add")

        self.lin_rel = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_root = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        out = out + self.lin_root(x)
        return out

    def message(self, x_j):
        return self.lin_rel(x_j)


# ----------------------------------------------------------
# LigandGNN (MATCHES CHECKPOINT EXACTLY)
# ----------------------------------------------------------
class LigandGNN(nn.Module):
    def __init__(
        self,
        in_dim=32,
        hidden_dim=192,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # ----------- Conv stack -----------
        self.convs.append(RelConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(RelConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ----------- Head (IMPORTANT: name = head.net) -----------
        self.head = nn.Module()
        self.head.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ----------- Output layer -----------
        self.lin_out = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.head.net(x)
        return self.lin_out(x)
