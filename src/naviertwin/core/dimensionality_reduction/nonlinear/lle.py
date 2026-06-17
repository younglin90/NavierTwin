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

from naviertwin._native import HAS_NATIVE_KERNELS, _kernels


def _solve_dense(A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    mat = np.ascontiguousarray(A, dtype=np.float64)
    rhs = np.ascontiguousarray(b, dtype=np.float64)
    if HAS_NATIVE_KERNELS and _kernels is not None:
        try:
            return _kernels.solve_dense(mat, rhs)
        except Exception:
            pass
    return getattr(np.linalg, "solve")(mat, rhs)


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
    i = 0
    while i < n:
        Z = X[nbrs[i]] - X[i]
        C = Z @ Z.T
        C = C + reg * np.trace(C) * np.eye(k) / k
        w = _solve_dense(C, np.ones(k))
        w = w / w.sum()
        W[i, nbrs[i]] = w
        i += 1
    # M = (I - W).T (I - W)
    IW = np.eye(n) - W
    M = IW.T @ IW
    # 작은 eigenvalue 선택 (trivial 제외)
    evals, evecs = np.linalg.eigh(M)
    order = np.argsort(evals)
    return evecs[:, order[1:n_components + 1]]


__all__ = ["lle"]
