"""Fast-marching lite — 1D SDF reinit (Sethian style 단순화).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.geometry.sdf_reinit import fast_march_1d
    >>> phi = np.array([5., 3, 0.0, -3, -5])
    >>> phi2 = fast_march_1d(phi, dx=1.0)
    >>> phi2[2]
    0.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def fast_march_1d(
    phi: NDArray[np.float64], *, dx: float = 1.0,
) -> NDArray[np.float64]:
    """1D 등거리 격자 → |x - x_0|, x_0 = 가장 가까운 zero-crossing."""
    return _kernels.fast_march_1d(np.asarray(phi, dtype=np.float64), float(dx))


__all__ = ["fast_march_1d"]
