"""Moving Least Squares (MLS) — local weighted polynomial fit.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.meshless.mls import mls_eval_1d
    >>> x = np.linspace(0, 1, 11); y = x ** 2
    >>> mls_eval_1d(x_query=0.5, x=x, y=y, h=0.3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def mls_eval_1d(
    *, x_query: float, x: NDArray[np.float64], y: NDArray[np.float64],
    h: float = 0.1, order: int = 2,
) -> float:
    """Local poly-`order` MLS at x_query."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.exp(-((x - x_query) / h) ** 2)
    P = np.vander(x - x_query, order + 1, increasing=True)  # (n, order+1)
    A = (P * w[:, None]).T @ P
    b = (P * w[:, None]).T @ y
    coef = np.asarray(_kernels.solve_square(A + 1e-12 * np.eye(A.shape[0]), b), dtype=np.float64)
    return float(coef[0])  # value at center


__all__ = ["mls_eval_1d"]
