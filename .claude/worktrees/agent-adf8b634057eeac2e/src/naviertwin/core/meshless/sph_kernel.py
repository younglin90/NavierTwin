"""SPH cubic-spline kernel + density estimation (1D).

W(q) = σ * { 1 - 1.5 q² + 0.75 q³ if 0≤q<1; 0.25 (2-q)³ if 1≤q<2; 0 }.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.meshless.sph_kernel import cubic_spline_1d, density_1d
    >>> cubic_spline_1d(0.0, h=1.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def cubic_spline_1d(r: float | NDArray[np.float64], *, h: float = 1.0) -> NDArray[np.float64]:
    """Monaghan 1992 1D cubic-spline kernel (sigma_1D = 2/3)."""
    q = np.abs(np.asarray(r, dtype=np.float64)) / h
    sigma = 2.0 / 3.0 / h
    out = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    out = np.where(m1, sigma * (1.0 - 1.5 * q ** 2 + 0.75 * q ** 3), out)
    out = np.where(m2, sigma * 0.25 * (2.0 - q) ** 3, out)
    return out


def density_1d(
    positions: NDArray[np.float64],
    masses: NDArray[np.float64],
    *,
    h: float = 1.0,
) -> NDArray[np.float64]:
    return _kernels.sph_density_1d(
        np.asarray(positions, dtype=np.float64),
        np.asarray(masses, dtype=np.float64),
        float(h),
    )


__all__ = ["cubic_spline_1d", "density_1d"]
