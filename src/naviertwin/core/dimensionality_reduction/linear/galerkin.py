"""Galerkin projection — 선형/이차 연산자를 POD 모드공간으로 투영.

M ẋ = L x + N(x, x) 형태의 시스템을 Φᵀ M Φ · a_t = Φᵀ L Φ · a + Φᵀ N(Φa, Φa) 로.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.galerkin import (
    ...     project_linear_operator,
    ... )
    >>> Phi = np.eye(5)[:, :2]
    >>> L = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> L_red = project_linear_operator(Phi, L)
    >>> L_red.shape
    (2, 2)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def project_linear_operator(
    Phi: NDArray[np.float64], L: NDArray[np.float64],
) -> NDArray[np.float64]:
    """L_red = Φᵀ L Φ."""
    return Phi.T @ L @ Phi


def project_mass_matrix(
    Phi: NDArray[np.float64], M: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """M_red = Φᵀ M Φ (M=None 이면 identity)."""
    if M is None:
        return Phi.T @ Phi
    return Phi.T @ M @ Phi


def project_quadratic(
    Phi: NDArray[np.float64],
    N: Callable[[NDArray, NDArray], NDArray],
    a: NDArray[np.float64],
) -> NDArray[np.float64]:
    """축소 공간 계수 a → full 재구성 → 비선형 N(x,x) → Φᵀ 투영."""
    x = Phi @ a
    return Phi.T @ N(x, x)


def rom_rhs_linear(
    Phi: NDArray[np.float64],
    L: NDArray[np.float64],
    M: NDArray[np.float64] | None = None,
) -> Callable[[float, NDArray], NDArray]:
    """선형 ROM RHS factory: a → M_red⁻¹ L_red a."""
    L_red = project_linear_operator(Phi, L)
    M_red = project_mass_matrix(Phi, M)
    M_inv_L = np.asarray(_kernels.solve_square(M_red, L_red), dtype=np.float64)

    def rhs(t: float, a: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: ARG001
        return M_inv_L @ a

    return rhs


def project_field_to_modes(
    Phi: NDArray[np.float64], x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """full-order 상태 x → 계수 a = Φᵀ x (직교 Φ 가정)."""
    return Phi.T @ x


def reconstruct_from_modes(
    Phi: NDArray[np.float64], a: NDArray[np.float64],
) -> NDArray[np.float64]:
    """계수 → full-order 재구성."""
    return Phi @ a


__all__ = [
    "project_linear_operator",
    "project_mass_matrix",
    "project_quadratic",
    "rom_rhs_linear",
    "project_field_to_modes",
    "reconstruct_from_modes",
]
