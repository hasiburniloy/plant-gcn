"""Plant-GCN and a GAT variant for node regression on a spectral k-NN graph.

Each node is one preprocessed spectrum (paper section 2.6):

    gcn   GCNConv(in,h1) -> ReLU -> GCNConv(h1,h2) -> ReLU -> Dropout -> Linear(h2,out)
    gat   GATConv(in,h1,heads=4) -> LeakyReLU(0.2) -> Dropout -> GATConv(h1*4,h2,heads=4)
          -> LeakyReLU -> Dropout -> Linear(h2*4,out)

With return_hidden=True the intermediate activations are kept in `model.hidden`
(for later interpretability work); off by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class PlantGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden1: int = 64, hidden2: int = 32,
                 dropout: float = 0.3, return_hidden: bool = False) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(hidden2, out_dim)
        self.return_hidden = return_hidden
        self.hidden: dict = {}

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        if self.return_hidden:
            self.hidden = {"conv1": h1.detach(), "conv2": h2.detach()}
        return self.out(self.dropout(h2))


class GATModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden1: int = 64, hidden2: int = 32,
                 dropout: float = 0.3, heads: int = 4, return_hidden: bool = False) -> None:
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden1, heads=heads)
        self.gat2 = GATConv(hidden1 * heads, hidden2, heads=heads)
        self.out = nn.Linear(hidden2 * heads, out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.LeakyReLU(0.2)
        self.return_hidden = return_hidden
        self.hidden: dict = {}

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h1 = self.dropout(self.act(self.gat1(x, edge_index)))
        h2 = self.dropout(self.act(self.gat2(h1, edge_index)))
        if self.return_hidden:
            self.hidden = {"gat1": h1.detach(), "gat2": h2.detach()}
        return self.out(h2)


MODELS = {"gcn": PlantGCN, "gat": GATModel}


def build_model(name: str, in_dim: int, out_dim: int, **params) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"model.name must be one of {list(MODELS)}, got {name!r}")
    return MODELS[name](in_dim, out_dim, **params)
