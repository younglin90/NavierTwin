"""Balanced POD (Willcox/Peraire) — 직접 Gramians + balanced truncation.

linear discrete system (A, B, C):
- controllability Gramian: W_c = Σ A^k B Bᵀ (A^k)ᵀ
- observability Gramian: W_o = Σ (A^k)ᵀ Cᵀ C A^k

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.bpod_scratch import (
    ...     bpod_reduce,
    ... )
    >>> A = np.diag([0.9, 0.8, 0.7, 0.6])
    >>> B = np.array([[1.], [1.], [1.], [1.]])
    >>> C = np.array([[1., 1., 0., 0.]])
    >>> Ar, Br, Cr, T, Tinv = bpod_reduce(A, B, C, r=2)
    >>> Ar.shape
    (2, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray


def lyapunov_disc(
    A: NDArray[np.float64], Q: NDArray[np.float64], *, max_iter: int = 2000,
    tol: float = 1e-10,
) -> NDArray[np.float64]:
    """discrete Lyapunov: X = A X Aᵀ + Q, 반복 누적."""
    X = np.zeros_like(Q)
    term = Q.copy()
    iteration = 0
    while iteration < max_iter:
        X = X + term
        term = A @ term @ A.T
        if np.max(np.abs(term)) < tol:
            break
        iteration += 1
    return X


def bpod_reduce(
    A: NDArray[np.float64], B: NDArray[np.float64], C: NDArray[np.float64],
    r: int = 2,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """(A_r, B_r, C_r, T, T⁻¹) — balanced truncation."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    Wc = lyapunov_disc(A, B @ B.T)
    Wo = lyapunov_disc(A.T, C.T @ C)
    # Cholesky
    Lc = np.linalg.cholesky(Wc + 1e-12 * np.eye(Wc.shape[0]))
    Lo = np.linalg.cholesky(Wo + 1e-12 * np.eye(Wo.shape[0]))
    U, s, Vt = _svd(Lo.T @ Lc, full_matrices=False)
    k = int(min(r, s.size))
    s_sq = np.sqrt(s[:k])
    T = Lc @ Vt.T[:, :k] @ np.diag(1.0 / s_sq)
    Tinv = np.diag(1.0 / s_sq) @ U[:, :k].T @ Lo.T
    Ar = Tinv @ A @ T
    Br = Tinv @ B
    Cr = C @ T
    return Ar, Br, Cr, T, Tinv


__all__ = ["lyapunov_disc", "bpod_reduce"]
