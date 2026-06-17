"""Mutual information — kNN entropy estimator (Kraskov-lite, 1D-1D).

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.mi_knn import mi_knn_1d
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(200)
    >>> mi_knn_1d(x, x) > mi_knn_1d(x, rng.standard_normal(200))
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _entropy_knn(x: NDArray, k: int = 3) -> float:
    """Kozachenko-Leonenko entropy estimator (1D)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    # k-NN distance at every sample (max difference among k nearest).
    idx = k if k < n else -1
    d = np.sort(np.abs(x[:, None] - x[None, :]), axis=1)[:, idx]
    d = np.maximum(d, 1e-12)
    return float(np.mean(np.log(2 * d)) + np.log(n - 1) - np.log(k) + 0.5772)


def mi_knn_1d(x: NDArray[np.float64], y: NDArray[np.float64], *, k: int = 3) -> float:
    """MI(X;Y) ≈ H(X) + H(Y) - H(X,Y); 1D X, 1D Y."""
    Hx = _entropy_knn(x, k)
    Hy = _entropy_knn(y, k)
    # joint: project onto sum coordinate
    z = np.column_stack([np.asarray(x), np.asarray(y)])
    # crude joint entropy via 2D nearest neighbour
    n = z.shape[0]
    idx = k if k < n else -1
    dist = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    d = np.maximum(np.sort(dist, axis=1)[:, idx], 1e-12)
    Hxy = float(2 * np.mean(np.log(d)) + np.log(n - 1) - np.log(k) + np.log(np.pi))
    return float(Hx + Hy - Hxy)


__all__ = ["mi_knn_1d"]
