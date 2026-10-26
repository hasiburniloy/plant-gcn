"""Spectral k-nearest-neighbour graph (paper section 2.6).

Nodes are preprocessed spectra; edges join each node to its k nearest neighbours
(Euclidean, self excluded) and are made undirected. GCNConv adds self loops itself.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


def build_knn_edges(X: np.ndarray, k: int) -> torch.Tensor:
    """Undirected, de-duplicated edge_index (2, E) with both directions of every k-NN pair."""
    n = len(X)
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)
    k = min(k, n - 1)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    # drop the self column by value (duplicate spectra could otherwise put self elsewhere)
    src, dst = [], []
    for i in range(n):
        nbrs = [j for j in idx[i] if j != i][:k]
        src.extend([i] * len(nbrs))
        dst.extend(nbrs)
    src, dst = np.array(src), np.array(dst)
    both = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    both = np.unique(both, axis=1)
    return torch.tensor(both, dtype=torch.long)


def build_graph(X: np.ndarray, Y: np.ndarray, k: int) -> Data:
    """X: (N, bands) preprocessed spectra; Y: (N, n_targets) scaled targets."""
    return Data(x=torch.tensor(X, dtype=torch.float), edge_index=build_knn_edges(X, k),
                y=torch.tensor(Y, dtype=torch.float))


def append_nodes(ref_X: np.ndarray, ref_Y: np.ndarray, new_X: np.ndarray,
                 k: int) -> Tuple[Data, range]:
    """Rebuild the k-NN graph over [ref_X; new_X]; return it and the index range of new nodes.

    Used for test/new-data prediction. Labels of new nodes are unknown and set to zero;
    they are never used.
    """
    X = np.vstack([ref_X, new_X])
    Y = np.vstack([ref_Y, np.zeros((len(new_X), ref_Y.shape[1]))])
    return build_graph(X, Y, k), range(len(ref_X), len(X))
