"""Refinement criteria library — gradient / curvature / Richardson.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.amr.refine_criteria import gradient_indicator
    >>> u = np.array([0., 0, 1., 1., 1.])
    >>> gradient_indicator(u, dx=1.0)
    array([0. , 0.5, 0.5, 0. , 0. ])
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gradient_indicator(
    u: NDArray[np.float64], *, dx: float = 1.0,
) -> NDArray[np.float64]:
    u = np.asarray(u, dtype=np.float64)
    g = np.zeros_like(u)
    g[1:-1] = np.abs((u[2:] - u[:-2]) / (2 * dx))
    return g


def curvature_indicator(
    u: NDArray[np.float64], *, dx: float = 1.0,
) -> NDArray[np.float64]:
    u = np.asarray(u, dtype=np.float64)
    c = np.zeros_like(u)
    c[1:-1] = np.abs((u[2:] - 2 * u[1:-1] + u[:-2]) / (dx * dx))
    return c


def richardson_indicator(
    u_coarse: NDArray[np.float64], u_fine: NDArray[np.float64],
) -> NDArray[np.float64]:
    """|u_fine - u_coarse| (interp at coarse points)."""
    return np.abs(np.asarray(u_fine) - np.asarray(u_coarse))


def mark_refine(
    indicator: NDArray[np.float64], *, threshold: float = 0.1,
) -> NDArray[np.bool_]:
    return np.asarray(indicator) > threshold


__all__ = [
    "curvature_indicator", "gradient_indicator",
    "mark_refine", "richardson_indicator",
]
