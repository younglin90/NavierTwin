"""Weighted POD — 가중치 inner product <x,y>_W = xᵀ W y (e.g. mass matrix).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.weighted_pod import (
    ...     weighted_pod,
    ... )
    >>> X = np.random.default_rng(0).standard_normal((30, 10))
    >>> W = np.eye(30)
    >>> modes, sv = weighted_pod(X, W, k=3)
    >>> modes.shape
    (30, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray


def weighted_pod(
    X: NDArray[np.float64], W: NDArray[np.float64] | None = None, k: int = 5,
    *, center: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """weighted POD — W 는 SPD. modes, singular values.

    C = Xᵀ W X → eigh → modes = X V / σ.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X (n_features, n_snapshots)")
    if center:
        X = X - X.mean(axis=1, keepdims=True)
    if W is None:
        W = np.eye(X.shape[0])
    W = np.asarray(W, dtype=np.float64)
    N = X.shape[1]
    C = (X.T @ W @ X) / max(N - 1, 1)
    ev, vc = np.linalg.eigh(C)
    order = np.argsort(-ev)
    ev = np.maximum(ev[order], 0.0)
    vc = vc[:, order]
    k = int(min(k, N))
    sv = np.sqrt(ev[:k] * max(N - 1, 1))
    modes = X @ vc[:, :k]
    modes = modes / (sv + 1e-30)
    return modes, sv


def compressed_pod(
    X: NDArray[np.float64], k: int = 5, compress: int = 20,
    *, seed: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """cPOD — random sketch projection."""
    rng = np.random.default_rng(seed)
    n, N = X.shape
    compress = int(min(compress, n))
    S = rng.standard_normal((compress, n))
    Y = S @ X  # (compress, N)
    # SVD on smaller matrix
    _U_s, sv, Vt = _svd(Y, full_matrices=False)
    # reconstruct approximate full modes
    k = int(min(k, sv.size))
    # modes = X V / σ
    modes = X @ Vt.T[:, :k]
    # normalize
    norms = np.linalg.norm(modes, axis=0) + 1e-30
    modes = modes / norms
    return modes, sv[:k]


__all__ = ["weighted_pod", "compressed_pod"]
