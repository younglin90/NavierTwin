"""Locally Linear Embedding (Roweis & Saul 2000).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.lle import lle
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 3))
    >>> Y = lle(X, k=10, n_components=2)
    >>> Y.shape
    (100, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lle(
    X: NDArray[np.float64], k: int = 10, n_components: int = 2,
    *, reg: float = 1e-3,
) -> NDArray[np.float64]:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    # neighbors
    nbrs = np.argpartition(D, k, axis=1)[:, :k]
    # weights
    W = np.zeros((n, n))
    for i in range(n):
        Z = X[nbrs[i]] - X[i]
        C = Z @ Z.T
        C = C + reg * np.trace(C) * np.eye(k) / k
        w = np.linalg.solve(C, np.ones(k))
        w = w / w.sum()
        W[i, nbrs[i]] = w
    # M = (I - W).T (I - W)
    IW = np.eye(n) - W
    M = IW.T @ IW
    # 작은 eigenvalue 선택 (trivial 제외)
    evals, evecs = np.linalg.eigh(M)
    order = np.argsort(evals)
    return evecs[:, order[1:n_components + 1]]


__all__ = ["lle"]
