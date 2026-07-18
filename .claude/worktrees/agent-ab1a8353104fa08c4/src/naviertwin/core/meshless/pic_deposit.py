"""PIC (particle-in-cell) deposition — linear (CIC) charge to grid 1D.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.meshless.pic_deposit import deposit_cic_1d
    >>> x = np.array([0.5, 1.7])
    >>> rho = deposit_cic_1d(x, np.array([1.0, 1.0]), n_grid=4, dx=1.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def deposit_cic_1d(
    x: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    n_grid: int,
    dx: float = 1.0,
    x0: float = 0.0,
) -> NDArray[np.float64]:
    """Cloud-in-cell linear deposition."""
    return _kernels.deposit_cic_1d(
        np.asarray(x, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        int(n_grid),
        float(dx),
        float(x0),
    )


__all__ = ["deposit_cic_1d"]
