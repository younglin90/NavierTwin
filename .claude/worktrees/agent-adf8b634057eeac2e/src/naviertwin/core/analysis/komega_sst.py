"""k-ω SST (Menter 1994) — blended k-ω / k-ε.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.komega_sst import (
    ...     eddy_viscosity_sst, sst_blending_F1,
    ... )
    >>> k = np.array([1.0]); omega = np.array([1.0])
    >>> S = np.array([0.5])
    >>> nu_t = eddy_viscosity_sst(k, omega, S, F2=np.array([0.5]))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

A1 = 0.31


def eddy_viscosity_sst(
    k: NDArray[np.float64],
    omega: NDArray[np.float64],
    strain_rate_mag: NDArray[np.float64],
    F2: NDArray[np.float64],
    *,
    a1: float = A1,
) -> NDArray[np.float64]:
    """ν_t = a1 k / max(a1 ω, S F2)."""
    k = np.asarray(k)
    omega = np.asarray(omega)
    S = np.asarray(strain_rate_mag)
    F2 = np.asarray(F2)
    denom = np.maximum(a1 * omega, S * F2) + 1e-30
    return a1 * k / denom


def sst_blending_F1(
    k: NDArray[np.float64], omega: NDArray[np.float64], y: NDArray[np.float64],
    nu: float = 1e-5, CDkw: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """F1 blending function (Menter 2003)."""
    k = np.asarray(k)
    omega = np.asarray(omega)
    y = np.asarray(y)
    arg1 = np.minimum(
        np.maximum(np.sqrt(k) / (0.09 * omega * y + 1e-30), 500 * nu / (y * y * omega + 1e-30)),
        4 * 0.856 * k / ((CDkw if CDkw is not None else 1e-10) * y * y + 1e-30),
    )
    return np.tanh(arg1 ** 4)


__all__ = ["eddy_viscosity_sst", "sst_blending_F1", "A1"]
