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

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def arnoldi(
    A: NDArray[np.float64], b: NDArray[np.float64], k: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """k-step Arnoldi. 반환: (Q (n, k+1), H (k+1, k))."""
    Q, H = _kernels.arnoldi(np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64).ravel(), int(k))
    return np.asarray(Q, dtype=np.float64), np.asarray(H, dtype=np.float64)


def ritz_values(
    A: NDArray[np.float64], b: NDArray[np.float64], k: int,
) -> NDArray[np.complex128]:
    """H 의 eigenvalues = Ritz 추정."""
    _, H = arnoldi(A, b, k)
    kk = H.shape[1]
    return np.linalg.eigvals(H[:kk, :kk])


__all__ = ["arnoldi", "ritz_values"]
