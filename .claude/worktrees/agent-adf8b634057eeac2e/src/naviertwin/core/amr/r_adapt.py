"""r-adaptation — node movement toward high-gradient regions (1D).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.amr.r_adapt import r_adapt_1d
    >>> x = np.linspace(0, 1, 11)
    >>> w = np.where(x > 0.5, 5.0, 1.0)
    >>> x2 = r_adapt_1d(x, w, n_iter=20)
    >>> x2.shape
    (11,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def r_adapt_1d(
    x: NDArray[np.float64], weights: NDArray[np.float64], *, n_iter: int = 20,
) -> NDArray[np.float64]:
    """equidistribution: ∫ w dx between nodes equal."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by r_adapt_1d")
    return _kernels.r_adapt_1d(
        np.asarray(x, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        n_iter,
    )


__all__ = ["r_adapt_1d"]
