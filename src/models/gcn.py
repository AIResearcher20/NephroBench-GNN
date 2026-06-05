import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNModel(nn.Module):
    """
    Graph Convolutional Network (GCN)
    for molecular property prediction (nephrotoxicity classification)

    Reference:
    Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
    """
    def __init__(self, input_dim=25, hidden_dim=128, num_layers=3, dropout=0.5):
        super(GCNModel, self).__init__()

        self.convs = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Output layer
        x = torch.sigmoid(self.lin(x)).view(-1)

        return x
