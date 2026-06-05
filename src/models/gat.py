import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) for molecular property prediction.

    Reference:
    Veličković et al. (2018) - Graph Attention Networks
    """
    def __init__(self, input_dim=25, hidden_dim=128, heads=4, dropout=0.5):
        super(GATModel, self).__init__()

        # First GAT layer (multi-head attention)
        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout
        )

        # Second layer
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            concat=True,
            dropout=dropout
        )

        # Third layer
        self.conv3 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.elu(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        x = self.dropout(x)
        x = torch.sigmoid(self.classifier(x)).view(-1)

        return x
