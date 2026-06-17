"""Adjoint 민감도 — dJ/dp via adjoint (선형 시스템 기반).

선형 방정식 A(p) u = b(p) 와 스칼라 목적 J(u, p) 에 대해
dJ/dp = ∂J/∂p - λᵀ (∂A/∂p · u - ∂b/∂p), λ = A⁻ᵀ ∂J/∂u.

Examples:
    >>> # 간단한 quadratic 예제는 test 참조
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def linear_adjoint_sensitivity(
    A_fn: Callable[[NDArray], NDArray],
    b_fn: Callable[[NDArray], NDArray],
    J_fn: Callable[[NDArray, NDArray], float],
    p: NDArray[np.float64],
    *,
    eps: float = 1e-6,
) -> tuple[float, NDArray[np.float64]]:
    """A(p) u = b(p) 시스템과 J(u, p) 의 dJ/dp (FD Jacobian).

    Returns:
        (J 값, dJ/dp).
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    A = A_fn(p)
    b = b_fn(p)
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by adjoint")
    u = _kernels.solve_dense(A, b)
    J0 = float(J_fn(u, p))

    # ∂J/∂u (FD)
    dJdu = np.zeros_like(u)
    i = 0
    while i < u.size:
        up = u.copy()
        up[i] += eps
        dJdu[i] = (J_fn(up, p) - J0) / eps
        i += 1

    # adjoint: Aᵀ λ = ∂J/∂u
    lam = _kernels.solve_dense(A.T, dJdu)

    # dJ/dp = ∂J/∂p - λᵀ (∂A/∂p u - ∂b/∂p)  (FD on A, b)
    grad = np.zeros_like(p)
    k = 0
    while k < p.size:
        pp = p.copy()
        pp[k] += eps
        dA_u = (A_fn(pp) - A) @ u / eps
        db = (b_fn(pp) - b) / eps
        # ∂J/∂p (explicit)
        dJdp_explicit = (J_fn(u, pp) - J0) / eps
        grad[k] = dJdp_explicit - lam @ (dA_u - db)
        k += 1
    return J0, grad


def fd_sensitivity(
    forward: Callable[[NDArray], float],
    p: NDArray[np.float64], *, eps: float = 1e-6,
) -> tuple[float, NDArray[np.float64]]:
    """순수 FD (adjoint 결과 검증용)."""
    p = np.asarray(p, dtype=np.float64).ravel()
    J0 = float(forward(p))
    g = np.zeros_like(p)
    i = 0
    while i < p.size:
        pp = p.copy()
        pp[i] += eps
        g[i] = (forward(pp) - J0) / eps
        i += 1
    return J0, g


__all__ = ["linear_adjoint_sensitivity", "fd_sensitivity"]
