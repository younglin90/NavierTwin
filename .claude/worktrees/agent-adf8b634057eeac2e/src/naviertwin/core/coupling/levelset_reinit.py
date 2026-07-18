"""Level-set reinitialization (1D, Sussman-Smereka 1994).

φ_τ + sign(φ_0)(|∇φ| - 1) = 0  → 부호 거리 함수.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.coupling.levelset_reinit import reinit_1d
    >>> phi = np.array([2., 1., 0.5, -0.5, -1., -2.])
    >>> phi2 = reinit_1d(phi, dx=1.0, n_iter=20)
    >>> phi2.shape
    (6,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def reinit_1d(
    phi: NDArray[np.float64], *, dx: float = 1.0, n_iter: int = 30,
) -> NDArray[np.float64]:
    return _kernels.levelset_reinit_1d(
        np.asarray(phi, dtype=np.float64),
        float(dx),
        int(n_iter),
    )


__all__ = ["reinit_1d"]
