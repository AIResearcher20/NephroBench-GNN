"""
GIN Model for Molecular Property Prediction

Graph Isomorphism Network (GIN) implementation
for nephrotoxicity prediction using PyTorch Geometric.

Author: Vania Karimi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class GINModel(nn.Module):
    """
    Graph Isomorphism Network (GIN)

    Parameters
    ----------
    input_dim : int
        Number of atom-level input features (default: 25)

    hidden_dim : int
        Hidden embedding dimension (default: 128)

    dropout : float
        Dropout probability (default: 0.7)
    """

    def __init__(
        self,
        input_dim: int = 25,
        hidden_dim: int = 128,
        dropout: float = 0.7
    ):
        super(GINModel, self).__init__()

        # First GIN convolution block
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Second GIN convolution block
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

        # Graph-level classifier
        self.classifier = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)


    def forward(self, data):
        """
        Forward pass

        Parameters
        ----------
        data : torch_geometric.data.Data
            Molecular graph batch

        Returns
        -------
        torch.Tensor
            Probability predictions
        """

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # Message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph-level representation
        x = global_mean_pool(x, batch)

        # Regularization
        x = self.dropout(x)

        # Binary classification probability
        x = torch.sigmoid(self.classifier(x))

        return x.view(-1)
