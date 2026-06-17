"""Nelder-Mead simplex (1965).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.nelder_mead import nelder_mead
    >>> x = nelder_mead(lambda x: float(x @ x), x0=np.array([1.0, 1.0]))
    >>> np.linalg.norm(x) < 0.01
    True
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def nelder_mead(
    f: Callable[[NDArray], float],
    x0: NDArray[np.float64],
    *,
    step: float = 0.5,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    n = len(x0)
    x0_arr = np.asarray(x0, dtype=np.float64)
    simplex = np.repeat(x0_arr[None, :], n + 1, axis=0)
    simplex[1 + np.arange(n), np.arange(n)] += step
    fvals = np.fromiter(map(f, simplex), dtype=np.float64, count=n + 1)
    iter_idx = 0
    while iter_idx < max_iter:
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        if np.linalg.norm(np.asarray(simplex[-1]) - simplex[0]) < tol:
            break
        iter_idx += 1
        centroid = np.mean(simplex[:-1], axis=0)
        # reflect
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
            continue
        if fr < fvals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
            continue
        # contract
        xc = centroid + 0.5 * (simplex[-1] - centroid)
        fc = f(xc)
        if fc < fvals[-1]:
            simplex[-1] = xc
            fvals[-1] = fc
            continue
        # shrink
        simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
        fvals[1:] = np.fromiter(map(f, simplex[1:]), dtype=np.float64, count=n)
    return simplex[0]


__all__ = ["nelder_mead"]
