# model.py
"""
LigandGNN - simple GNN encoder + MLP head.
Provides init_output_bias(val) helper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class LigandGNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=192, dropout=0.15, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        cur_dim = node_feature_dim
        for i in range(n_layers):
            self.convs.append(GraphConv(cur_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            cur_dim = hidden_dim

        self.dropout = dropout
        self.head = MLP(hidden_dim, hidden_dim, hidden_dim, n_layers=2, dropout=dropout)
        self.lin_out = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        g = global_mean_pool(x, batch)
        h = self.head(g)
        out = self.lin_out(h).view(-1, 1)
        return out.squeeze(-1)

    def init_output_bias(self, val: float):
        with torch.no_grad():
            try:
                self.lin_out.bias.fill_(float(val))
            except Exception:
                pass
