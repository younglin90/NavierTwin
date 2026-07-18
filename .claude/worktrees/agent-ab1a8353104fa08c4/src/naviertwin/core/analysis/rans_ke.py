"""RANS k-ε turbulence — eddy viscosity ν_t = C_μ k²/ε.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.rans_ke import eddy_viscosity_ke
    >>> k = np.array([1.0, 0.5])
    >>> eps = np.array([0.1, 0.05])
    >>> nu_t = eddy_viscosity_ke(k, eps)
    >>> nu_t.shape
    (2,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

C_MU = 0.09


def eddy_viscosity_ke(
    k: NDArray[np.float64], eps: NDArray[np.float64], *, c_mu: float = C_MU,
) -> NDArray[np.float64]:
    """ν_t = C_μ k²/ε."""
    k = np.asarray(k, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    return c_mu * (k * k) / (eps + 1e-30)


def production_term(
    nu_t: NDArray[np.float64], strain_rate: NDArray[np.float64],
) -> NDArray[np.float64]:
    """P = 2 ν_t S_ij S_ij."""
    return 2.0 * nu_t * np.sum(strain_rate ** 2, axis=tuple(range(1, strain_rate.ndim)))


__all__ = ["eddy_viscosity_ke", "production_term", "C_MU"]
