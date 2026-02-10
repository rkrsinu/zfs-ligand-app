import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool

class LigandEncoder(nn.Module):
    def __init__(self, node_dim, hidden=128, layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        d = node_dim
        for _ in range(layers):
            self.convs.append(GraphConv(d, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
            d = hidden

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        return global_mean_pool(x, batch)
