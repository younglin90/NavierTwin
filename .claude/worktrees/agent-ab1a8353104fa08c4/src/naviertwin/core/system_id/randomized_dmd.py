"""Randomized DMD — Erichson et al. 2019.

Sketch X1 by random Gaussian → QR → standard DMD on sketched.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.system_id.randomized_dmd import randomized_dmd
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 30))
    >>> evals, modes = randomized_dmd(X, rank=4)
    >>> modes.shape
    (50, 4)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray


def randomized_dmd(
    X: NDArray[np.float64],
    *,
    rank: int = 5,
    oversamp: int = 5,
    seed: int = 0,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Randomized DMD; 반환: (eigenvalues, modes)."""
    X = np.asarray(X, dtype=np.float64)
    n, m = X.shape
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    rng = np.random.default_rng(seed)
    p = rank + oversamp
    Omega = rng.standard_normal((m - 1, p))
    Y = X1 @ Omega  # (n, p)
    Q, _ = np.linalg.qr(Y)
    B1 = Q.T @ X1  # (p, m-1)
    B2 = Q.T @ X2
    U, s, Vt = _svd(B1, full_matrices=False)
    r = min(rank, U.shape[1])
    U = U[:, :r]
    s = s[:r]
    Vt = Vt[:r]
    A_tilde = U.T @ B2 @ Vt.T @ np.diag(1.0 / s)
    evals, W = np.linalg.eig(A_tilde)
    modes = Q @ U @ W  # lift back
    return evals, modes


__all__ = ["randomized_dmd"]
