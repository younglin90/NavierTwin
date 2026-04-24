"""Arnoldi iteration — Krylov 기저 + Hessenberg 행렬 + Ritz 값.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.arnoldi import arnoldi
    >>> A = np.diag([5.0, 4.0, 3.0, 2.0, 1.0])
    >>> Q, H = arnoldi(A, np.ones(5), k=3)
    >>> Q.shape, H.shape
    ((5, 4), (4, 3))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def arnoldi(
    A: NDArray[np.float64], b: NDArray[np.float64], k: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """k-step Arnoldi. 반환: (Q (n, k+1), H (k+1, k))."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    k = int(min(k, n))
    Q = np.zeros((n, k + 1))
    H = np.zeros((k + 1, k))
    q = np.asarray(b, dtype=np.float64).ravel()
    q = q / (np.linalg.norm(q) + 1e-30)
    Q[:, 0] = q
    for j in range(k):
        v = A @ Q[:, j]
        for i in range(j + 1):
            H[i, j] = Q[:, i] @ v
            v = v - H[i, j] * Q[:, i]
        H[j + 1, j] = float(np.linalg.norm(v))
        if H[j + 1, j] < 1e-14:
            return Q[:, :j + 1], H[:j + 1, :j + 1]
        Q[:, j + 1] = v / H[j + 1, j]
    return Q, H


def ritz_values(
    A: NDArray[np.float64], b: NDArray[np.float64], k: int,
) -> NDArray[np.complex128]:
    """H 의 eigenvalues = Ritz 추정."""
    _, H = arnoldi(A, b, k)
    kk = H.shape[1]
    return np.linalg.eigvals(H[:kk, :kk])


__all__ = ["arnoldi", "ritz_values"]
