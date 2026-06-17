"""음향 모드 분석 — 1D Helmholtz 고유 모드 + Strouhal/Womersley 유틸.

1D Helmholtz:
    d²p/dx² + k² p = 0,   p(0)=p(L)=0 (또는 dp/dx=0 at endpoints)

Dirichlet 해: p_n = sin(n π x / L), k_n = n π / L, f_n = c k_n / (2π).

Examples:
    >>> from naviertwin.core.flow_analysis.acoustic import (
    ...     duct_modes_dirichlet, strouhal, womersley,
    ... )
    >>> freqs, modes = duct_modes_dirichlet(L=1.0, c=340.0, n_modes=3, n_points=100)
    >>> abs(freqs[0] - 170.0) < 1.0  # c/(2L) = 170
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def duct_modes_dirichlet(
    L: float,
    c: float,
    n_modes: int = 5,
    n_points: int = 128,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Dirichlet 경계 1D duct 의 acoustic 고유모드.

    Returns:
        (frequencies[Hz], modes[n_points, n_modes]).
    """
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by duct_modes_dirichlet")
    return _kernels.duct_modes_dirichlet(L, c, n_modes, n_points)


def duct_modes_neumann(
    L: float,
    c: float,
    n_modes: int = 5,
    n_points: int = 128,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Neumann 경계 (닫힌-닫힌 or 열린-열린) 1D duct.

    cos 모드; k_n = n π / L, n = 0, 1, 2, ....
    """
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by duct_modes_neumann")
    return _kernels.duct_modes_neumann(L, c, n_modes, n_points)


def strouhal(f: float, L: float, U: float) -> float:
    """St = f L / U."""
    if U <= 0 or L <= 0:
        raise ValueError("L, U 는 양수")
    return f * L / U


def womersley(
    omega: float, R: float, nu: float
) -> float:
    """Womersley 수 α = R · sqrt(ω / ν).

    맥동 유동의 관성/점성 상대 강도. 심장 혈류 분석 등.
    """
    if R <= 0 or nu <= 0:
        raise ValueError("R, ν 는 양수")
    return R * np.sqrt(omega / nu)


__all__ = [
    "duct_modes_dirichlet",
    "duct_modes_neumann",
    "strouhal",
    "womersley",
]
