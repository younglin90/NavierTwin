"""경계층 분석 — δ99 / δ* / θ / 벽 전단 / u_τ / TKE.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.boundary_layer import delta99
    >>> y = np.linspace(0, 1, 100)
    >>> u = np.tanh(5*y) * 10.0
    >>> d = delta99(y, u, U_edge=10.0)
    >>> d > 0
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def delta99(
    y: NDArray[np.float64],
    u: NDArray[np.float64],
    U_edge: float,
) -> float:
    """δ₉₉: u=0.99·U∞ 위치 (선형 보간)."""
    target = 0.99 * U_edge
    y = np.asarray(y, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    order = np.argsort(y)
    y, u = y[order], u[order]
    return float(_kernels.delta99_scan(y, u, target))


def displacement_thickness(
    y: NDArray[np.float64],
    u: NDArray[np.float64],
    U_edge: float,
) -> float:
    """δ* = ∫₀^∞ (1 - u/U∞) dy."""
    integrand = 1.0 - np.asarray(u) / U_edge
    return float(np.trapezoid(integrand, np.asarray(y)))


def momentum_thickness(
    y: NDArray[np.float64],
    u: NDArray[np.float64],
    U_edge: float,
) -> float:
    """θ = ∫ (u/U∞)(1 - u/U∞) dy."""
    ur = np.asarray(u) / U_edge
    return float(np.trapezoid(ur * (1.0 - ur), np.asarray(y)))


def wall_shear_stress(
    y: NDArray[np.float64],
    u: NDArray[np.float64],
    mu: float,
) -> float:
    """τ_w = μ · du/dy|_{y=0} (front-diff)."""
    y = np.asarray(y, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    order = np.argsort(y)
    y, u = y[order], u[order]
    # 2-order one-sided (3-point)
    if len(y) < 3:
        return float(mu * (u[1] - u[0]) / (y[1] - y[0]))
    h1 = y[1] - y[0]
    h2 = y[2] - y[0]
    # u'(0) ≈ (-u2 h1² + u1 h2² - u0 (h2²-h1²)) / (h1 h2 (h2-h1))
    du_dy = (-u[2] * h1 ** 2 + u[1] * h2 ** 2 - u[0] * (h2 ** 2 - h1 ** 2)) / (
        h1 * h2 * (h2 - h1)
    )
    return float(mu * du_dy)


def friction_velocity(tau_w: float, rho: float) -> float:
    """u_τ = √(τ_w/ρ)."""
    return float(np.sqrt(abs(tau_w) / rho))


def turbulent_ke(
    u_prime: NDArray[np.float64],
    v_prime: NDArray[np.float64],
    w_prime: NDArray[np.float64] | None = None,
) -> float:
    """k = ½ <u'²+v'²+w'²>."""
    s = np.mean(u_prime ** 2) + np.mean(v_prime ** 2)
    if w_prime is not None:
        s += np.mean(w_prime ** 2)
    return float(0.5 * s)


__all__ = [
    "delta99", "displacement_thickness", "momentum_thickness",
    "wall_shear_stress", "friction_velocity", "turbulent_ke",
]
