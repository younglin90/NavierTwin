"""DEIM (Discrete Empirical Interpolation Method) — 비선형 항 hyperreduction.

Chaturantabut & Sorensen 2010.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.deim import deim
    >>> rng = np.random.default_rng(0)
    >>> U = np.linalg.qr(rng.standard_normal((20, 5)))[0]
    >>> P, idx = deim(U)
    >>> P.shape, idx.shape
    ((20, 5), (5,))
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


def deim(U: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """DEIM greedy index selection.

    Args:
        U: (n, m) basis (POD modes of nonlinear snapshots).

    Returns:
        P: (n, m) selection matrix (columns of identity).
        idx: (m,) selected DEIM indices.
    """
    U = np.asarray(U, dtype=np.float64)
    n, m = U.shape
    idx = np.zeros(m, dtype=int)
    # first index: argmax |u_1|
    idx[0] = int(np.argmax(np.abs(U[:, 0])))
    P = np.zeros((n, m))
    P[idx[0], 0] = 1.0
    j = 1
    while j < m:
        # solve (Pᵀ U[:, :j]) c = Pᵀ U[:, j]
        Pj = P[:, :j]
        Uj = U[:, :j]
        u_new = U[:, j]
        c = _solve_dense(Pj.T @ Uj, Pj.T @ u_new)
        r = u_new - Uj @ c
        idx[j] = int(np.argmax(np.abs(r)))
        P[idx[j], j] = 1.0
        j += 1
    return P, idx


def deim_project(
    U: NDArray[np.float64], P: NDArray[np.float64], f_at_idx: NDArray[np.float64],
) -> NDArray[np.float64]:
    """DEIM 보간: f ≈ U (Pᵀ U)⁻¹ f_at_idx."""
    PU = P.T @ U
    coef = _solve_dense(PU, f_at_idx)
    return U @ coef


__all__ = ["deim", "deim_project"]
