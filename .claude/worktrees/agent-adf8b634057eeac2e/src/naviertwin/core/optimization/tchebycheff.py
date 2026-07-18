"""Tchebycheff scalarization (multi-objective).

g_te(x) = max_i w_i |f_i(x) - z*_i|.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.tchebycheff import tchebycheff
    >>> f = np.array([1.0, 2.0])
    >>> w = np.array([0.5, 0.5])
    >>> z = np.array([0.0, 0.0])
    >>> tchebycheff(f, w, z)
    1.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def tchebycheff(
    f: NDArray[np.float64],
    weights: NDArray[np.float64],
    z_star: NDArray[np.float64],
) -> float:
    """g_te = max_i w_i |f_i - z*_i|."""
    return float(np.max(np.asarray(weights) * np.abs(np.asarray(f) - np.asarray(z_star))))


def weight_grid(n_obj: int = 2, n_weights: int = 5) -> NDArray[np.float64]:
    """Simple uniform weight vectors on the simplex (2D only)."""
    if n_obj == 2:
        w = np.linspace(0, 1, n_weights)
        return np.column_stack([w, 1 - w])
    raise NotImplementedError("only 2-objective grid implemented")


__all__ = ["tchebycheff", "weight_grid"]
