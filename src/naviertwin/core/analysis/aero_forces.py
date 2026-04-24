"""공력 계수 — Cp / Cd / Cl / 표면 힘 적분.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.aero_forces import pressure_coefficient
    >>> p = np.array([100., 150., 200.])
    >>> cp = pressure_coefficient(p, p_ref=100., rho=1.0, U_ref=10.)
    >>> cp.tolist()
    [0.0, 1.0, 2.0]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pressure_coefficient(
    p: NDArray[np.float64],
    p_ref: float,
    rho: float,
    U_ref: float,
) -> NDArray[np.float64]:
    """Cp = (p - p_ref) / (½ρU²)."""
    q = 0.5 * rho * U_ref ** 2
    if q <= 0:
        raise ValueError("dynamic pressure ½ρU² must be > 0")
    return (np.asarray(p, dtype=np.float64) - p_ref) / q


def surface_force(
    p: NDArray[np.float64],
    normals: NDArray[np.float64],
    areas: NDArray[np.float64],
) -> NDArray[np.float64]:
    """∫ p·n·dA. normals: (n, 3), 결과: (3,) 벡터."""
    p = np.asarray(p, dtype=np.float64).ravel()
    n = np.asarray(normals, dtype=np.float64)
    a = np.asarray(areas, dtype=np.float64).ravel()
    if not (p.shape == a.shape and n.shape[0] == p.size and n.shape[1] == 3):
        raise ValueError("shape 불일치")
    # pressure acts opposite to outward normal → force on body = -p n dA
    f = -(p * a)[:, None] * n
    return f.sum(axis=0)


def drag_lift_coefficients(
    force: NDArray[np.float64],
    rho: float,
    U_ref: float,
    area_ref: float,
    *,
    drag_dir: NDArray[np.float64] | None = None,
    lift_dir: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """(Cd, Cl). drag_dir/lift_dir 기본=(1,0,0)/(0,1,0)."""
    force = np.asarray(force, dtype=np.float64).ravel()
    if force.size != 3:
        raise ValueError("force must be (3,)")
    dd = np.asarray(drag_dir if drag_dir is not None else (1, 0, 0), dtype=np.float64)
    ll = np.asarray(lift_dir if lift_dir is not None else (0, 1, 0), dtype=np.float64)
    q = 0.5 * rho * U_ref ** 2 * area_ref
    if q <= 0:
        raise ValueError("reference q*A must be > 0")
    return {
        "drag": float(force @ dd),
        "lift": float(force @ ll),
        "Cd": float((force @ dd) / q),
        "Cl": float((force @ ll) / q),
    }


__all__ = ["pressure_coefficient", "surface_force", "drag_lift_coefficients"]
