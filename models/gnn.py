"""
Molecular Property Prediction GNN
----------------------------------
Graph Neural Network for predicting molecular properties (QM9 dataset).
Architecture: Message Passing Neural Network (MPNN) with global pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool
from torch_geometric.nn import BatchNorm


class MPNNLayer(nn.Module):
    """
    Single Message Passing layer using NNConv.
    Edge features are used to parameterize the message function.
    """
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        # Edge network: maps edge features -> weight matrix for node transform
        edge_nn = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, node_dim * out_dim)
        )
        self.conv = NNConv(node_dim, out_dim, edge_nn, aggr='mean')
        self.norm = BatchNorm(out_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.norm(x)
        return F.relu(x)


class MolecularGNN(nn.Module):
    """
    Full MPNN for molecular property prediction.

    Args:
        node_dim:    Input node feature dimension (11 for QM9)
        edge_dim:    Input edge feature dimension (4 for QM9)
        hidden_dim:  Hidden layer size
        num_layers:  Number of message passing layers
        num_targets: Number of output properties to predict
        pooling:     'mean' or 'sum' graph-level pooling
    """
    def __init__(
        self,
        node_dim: int = 11,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_targets: int = 1,
        pooling: str = 'mean'
    ):
        super().__init__()
        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Readout MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_targets)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # input features
        x = F.relu(self.input_proj(x))

        # Message passing
        for layer in self.mp_layers:
            x = layer(x, edge_index, edge_attr)

        # Graph-level pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_add_pool(x, batch)

        return self.readout(x)
