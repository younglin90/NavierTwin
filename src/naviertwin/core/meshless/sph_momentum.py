"""SPH momentum eq 1D — du_i/dt = -Σ m_j (p_i/ρ_i² + p_j/ρ_j²) ∂W/∂x.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.meshless.sph_momentum import sph_acceleration_1d
    >>> x = np.linspace(0, 1, 5); m = np.full(5, 0.2); rho = np.ones(5); p = np.ones(5)
    >>> a = sph_acceleration_1d(x, m, rho, p, h=0.3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def cubic_spline_grad_1d(r: NDArray[np.float64], *, h: float = 1.0) -> NDArray[np.float64]:
    q = np.abs(r) / h
    sigma = 2.0 / 3.0 / h
    sign = np.sign(r)
    out = np.zeros_like(r, dtype=np.float64)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    out = np.where(m1, sigma * sign * (-3.0 * q + 2.25 * q ** 2) / h, out)
    out = np.where(m2, sigma * sign * -0.75 * (2.0 - q) ** 2 / h, out)
    return out


def sph_acceleration_1d(
    x: NDArray[np.float64], m: NDArray[np.float64],
    rho: NDArray[np.float64], p: NDArray[np.float64], *,
    h: float = 1.0,
) -> NDArray[np.float64]:
    return _kernels.sph_acceleration_1d(
        np.asarray(x, dtype=np.float64),
        np.asarray(m, dtype=np.float64),
        np.asarray(rho, dtype=np.float64),
        np.asarray(p, dtype=np.float64),
        float(h),
    )


__all__ = ["cubic_spline_grad_1d", "sph_acceleration_1d"]
