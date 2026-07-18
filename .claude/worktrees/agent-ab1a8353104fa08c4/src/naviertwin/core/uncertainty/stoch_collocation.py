"""Stochastic collocation on sparse grid (Smolyak 1D-product, Clenshaw-Curtis).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.stoch_collocation import (
    ...     clenshaw_curtis_nodes, sparse_grid_2d,
    ... )
    >>> nodes, weights = clenshaw_curtis_nodes(level=3)
    >>> nodes.shape
    (5,)
"""

from __future__ import annotations

from math import comb

import numpy as np
from numpy.typing import NDArray


def clenshaw_curtis_nodes(level: int) -> tuple[NDArray, NDArray]:
    """level→ N=2^(level-1)+1 nodes on [-1,1]."""
    if level == 1:
        return np.array([0.0]), np.array([2.0])
    n = 2 ** (level - 1) + 1
    k = np.arange(n)
    x = np.cos(np.pi * k / (n - 1))
    # Clenshaw-Curtis weights
    j = np.arange(1, (n - 1) // 2 + 1, dtype=np.float64)
    c = np.full(n, 2.0)
    c[[0, -1]] = 1.0
    if j.size:
        weights = 2.0 / (4 * j * j - 1)
        angles = 2 * np.pi * np.outer(k, j) / (n - 1)
        s = np.cos(angles) @ weights
    else:
        s = np.zeros(n)
    w = c / (n - 1) * (1 - s)
    return x, w


def sparse_grid_2d(level: int = 3) -> tuple[NDArray, NDArray]:
    """2D Smolyak (sum_{l1+l2 <= level+1} ⊗) on Clenshaw-Curtis."""
    nodes_all = []
    weights_all = []
    l1 = 1
    while l1 <= level:
        l2 = 1
        while l2 < level + 2 - l1:
            x1, w1 = clenshaw_curtis_nodes(l1)
            x2, w2 = clenshaw_curtis_nodes(l2)
            X1, X2 = np.meshgrid(x1, x2, indexing="ij")
            W1, W2 = np.meshgrid(w1, w2, indexing="ij")
            sign = (-1) ** (level + 1 - (l1 + l2))
            coef = sign * comb(1, level + 1 - (l1 + l2)) if (level + 1 - (l1 + l2)) <= 1 else 0
            if coef != 0:
                pts = np.column_stack([X1.ravel(), X2.ravel()])
                ws = (W1 * W2).ravel() * coef
                nodes_all.append(pts)
                weights_all.append(ws)
            l2 += 1
        l1 += 1
    nodes = np.vstack(nodes_all)
    weights = np.concatenate(weights_all)
    return nodes, weights


__all__ = ["clenshaw_curtis_nodes", "sparse_grid_2d"]
