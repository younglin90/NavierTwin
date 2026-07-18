"""압축성 유동 유틸 — Mach, 음속, isentropic relations.

Examples:
    >>> from naviertwin.core.flow_analysis.thermofluids.compressible import (
    ...     speed_of_sound, mach_number, isentropic_p_ratio,
    ... )
    >>> a = speed_of_sound(gamma=1.4, R=287, T=300)
    >>> abs(a - 347.2) < 0.5
    True
    >>> M = mach_number(u=100.0, a=a)
    >>> abs(M - 0.288) < 0.01
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def speed_of_sound(
    gamma: float = 1.4,
    R: float = 287.0,
    T: float | NDArray[np.float64] = 293.15,
) -> float | NDArray[np.float64]:
    """이상 기체 음속 a = sqrt(γ R T)."""
    return np.sqrt(gamma * R * np.asarray(T))


def mach_number(
    u: float | NDArray[np.float64],
    a: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """M = |u| / a."""
    return np.abs(np.asarray(u)) / np.asarray(a)


def isentropic_p_ratio(
    M: float | NDArray[np.float64],
    gamma: float = 1.4,
) -> float | NDArray[np.float64]:
    """p_0 / p = (1 + (γ-1)/2 · M²)^{γ/(γ-1)}."""
    M = np.asarray(M)
    return (1.0 + (gamma - 1) / 2 * M ** 2) ** (gamma / (gamma - 1))


def isentropic_T_ratio(
    M: float | NDArray[np.float64],
    gamma: float = 1.4,
) -> float | NDArray[np.float64]:
    """T_0 / T = 1 + (γ-1)/2 · M²."""
    M = np.asarray(M)
    return 1.0 + (gamma - 1) / 2 * M ** 2


def isentropic_rho_ratio(
    M: float | NDArray[np.float64],
    gamma: float = 1.4,
) -> float | NDArray[np.float64]:
    """ρ_0 / ρ = (1 + (γ-1)/2 · M²)^{1/(γ-1)}."""
    M = np.asarray(M)
    return (1.0 + (gamma - 1) / 2 * M ** 2) ** (1.0 / (gamma - 1))


__all__ = [
    "speed_of_sound",
    "mach_number",
    "isentropic_p_ratio",
    "isentropic_T_ratio",
    "isentropic_rho_ratio",
]
