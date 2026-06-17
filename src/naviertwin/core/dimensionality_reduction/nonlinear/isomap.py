"""ISOMAP — k-NN graph + geodesic distance + MDS.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.isomap import isomap
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(0, 4*np.pi, 200)
    >>> X = np.stack([t*np.cos(t), t*np.sin(t)], axis=1)  # Swiss-roll-1D
    >>> Y = isomap(X, k=10, n_components=1)
    >>> Y.shape
    (200, 1)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _knn_graph(X: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    n = X.shape[0]
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    G = np.full_like(D, np.inf)
    idx = np.argpartition(D, k + 1, axis=1)[:, :k + 1]
    rows = np.arange(n)[:, np.newaxis]
    G[rows, idx] = D[rows, idx]
    # symmetrize
    G = np.minimum(G, G.T)
    return G


def _floyd_warshall(G: NDArray[np.float64]) -> NDArray[np.float64]:
    try:
        from scipy.sparse.csgraph import shortest_path

        return np.asarray(shortest_path(G, directed=False), dtype=np.float64)
    except Exception:
        pass
    n = G.shape[0]
    D = G.copy()
    k = 0
    while k < n:
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
        k += 1
    return D


def isomap(
    X: NDArray[np.float64], k: int = 5, n_components: int = 2,
) -> NDArray[np.float64]:
    """geodesic MDS."""
    X = np.asarray(X, dtype=np.float64)
    G = _knn_graph(X, k)
    D = _floyd_warshall(G)
    # MDS
    D2 = D ** 2
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(-evals)
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    return evecs[:, :n_components] * np.sqrt(evals[:n_components])


__all__ = ["isomap"]
