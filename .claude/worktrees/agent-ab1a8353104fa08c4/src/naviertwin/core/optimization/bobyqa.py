"""BOBYQA-lite — Powell 2009 spirit, simplified trust-region quadratic interp.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.bobyqa import bobyqa_lite
    >>> def f(x): return float((x[0] - 2.0) ** 2 + (x[1] + 1.0) ** 2)
    >>> x = bobyqa_lite(f, x0=np.zeros(2), max_iter=50)
    >>> np.allclose(x, [2.0, -1.0], atol=0.05)
    True
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def bobyqa_lite(
    f: Callable[[NDArray], float],
    x0: NDArray[np.float64],
    *,
    rho: float = 0.5,
    rho_min: float = 1e-4,
    max_iter: int = 100,
) -> NDArray[np.float64]:
    """간단 derivative-free trust region (axis search + shrink)."""
    x = np.asarray(x0, dtype=np.float64).copy()
    n = x.shape[0]
    fx = f(x)
    iter_idx = 0
    while iter_idx < max_iter:
        improved = False
        i = 0
        while i < n:
            sign_idx = 0
            signs = (+1, -1)
            while sign_idx < len(signs):
                trial = x.copy()
                trial[i] += signs[sign_idx] * rho
                ft = f(trial)
                if ft < fx:
                    fx = ft
                    x = trial
                    improved = True
                    break
                sign_idx += 1
            if improved:
                break
            i += 1
        if not improved:
            rho *= 0.5
            if rho < rho_min:
                break
        iter_idx += 1
    return x


__all__ = ["bobyqa_lite"]
