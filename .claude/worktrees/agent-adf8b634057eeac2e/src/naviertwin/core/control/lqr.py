"""LQR — discrete algebraic Riccati (iterative).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.control.lqr import dare, lqr_gain
    >>> A = np.array([[1.1, 0.1], [0.0, 1.0]])
    >>> B = np.array([[0.0], [0.1]])
    >>> Q = np.eye(2)
    >>> R = np.eye(1)
    >>> K = lqr_gain(A, B, Q, R)
    >>> K.shape
    (1, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def _solve_dense_rhs(A: NDArray[np.float64], B: NDArray[np.float64]) -> NDArray[np.float64]:
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by LQR")
    rhs = np.asarray(B, dtype=np.float64)
    if rhs.ndim == 1:
        return _kernels.solve_dense(A, rhs)
    out = np.empty_like(rhs)
    j = 0
    while j < rhs.shape[1]:
        out[:, j] = _kernels.solve_dense(A, rhs[:, j])
        j += 1
    return out


def dare(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
    *,
    n_iter: int = 200,
    tol: float = 1e-10,
) -> NDArray[np.float64]:
    """Discrete-time algebraic Riccati via iteration."""
    P = Q.copy().astype(np.float64)
    it = 0
    while it < n_iter:
        BtP = B.T @ P
        K = _solve_dense_rhs(R + BtP @ B, BtP @ A)
        P_new = A.T @ P @ (A - B @ K) + Q
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
        it += 1
    return P


def lqr_gain(
    A: NDArray, B: NDArray, Q: NDArray, R: NDArray,
) -> NDArray[np.float64]:
    """K = (R + Bᵀ P B)⁻¹ Bᵀ P A."""
    P = dare(A, B, Q, R)
    BtP = B.T @ P
    return _solve_dense_rhs(R + BtP @ B, BtP @ A)


__all__ = ["dare", "lqr_gain"]
