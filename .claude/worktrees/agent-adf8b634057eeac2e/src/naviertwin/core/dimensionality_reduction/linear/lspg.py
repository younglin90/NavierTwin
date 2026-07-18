"""LSPG (Least-Squares Petrov-Galerkin) projection.

서로 다른 trial Φ 와 test Ψ basis 로 residual 의 weighted least-squares 최소화.
선형 시스템 A x = b 에 대해 reduced system: (Ψᵀ A Φ) ẑ = Ψᵀ b.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.lspg import lspg_solve
    >>> A = np.diag([3.0, 2.0, 1.0])
    >>> b = np.array([3.0, 2.0, 1.0])
    >>> Phi = np.eye(3)[:, :2]
    >>> x = lspg_solve(A, b, Phi)
    >>> x.shape
    (3,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lspg_solve(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    Phi: NDArray[np.float64],
    *,
    Psi: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """LSPG: Ψ = A Φ (default). 반환 full-state x = Φ ẑ."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    Phi = np.asarray(Phi, dtype=np.float64)
    if Psi is None:
        Psi = A @ Phi
    Ar = Psi.T @ A @ Phi
    br = Psi.T @ b
    z_hat, *_ = np.linalg.lstsq(Ar, br, rcond=None)
    return Phi @ z_hat


def lspg_residual_norm(
    A: NDArray[np.float64], b: NDArray[np.float64], Phi: NDArray[np.float64],
) -> float:
    """LSPG 해의 residual 노름."""
    x = lspg_solve(A, b, Phi)
    return float(np.linalg.norm(A @ x - b))


__all__ = ["lspg_solve", "lspg_residual_norm"]
