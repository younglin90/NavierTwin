"""Closure-corrected ROM — fit residual closure term g(z) on truncation error.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.closure_rom import (
    ...     fit_closure,
    ... )
    >>> Z = np.array([[1.], [2.], [3.]])
    >>> R = np.array([[0.1], [0.2], [0.3]])
    >>> coef = fit_closure(Z, R)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fit_closure(
    Z: NDArray[np.float64], residual: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit linear closure: g(z) = K z. Least squares on (Z, R)."""
    Z = np.asarray(Z, dtype=np.float64)
    R = np.asarray(residual, dtype=np.float64)
    K, *_ = np.linalg.lstsq(Z, R, rcond=None)
    return K.T  # (out_dim, in_dim)


def apply_closure(K: NDArray, z: NDArray) -> NDArray:
    return K @ z


__all__ = ["apply_closure", "fit_closure"]
