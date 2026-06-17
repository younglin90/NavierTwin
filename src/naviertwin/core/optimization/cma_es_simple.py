"""간단 CMA-ES — isotropic 공분산만 (µ/λ).

완전한 CMA-ES 가 아닌 단순화: diag covariance, 상위 µ 평균.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.cma_es_simple import cma_es_simple
    >>> x, f = cma_es_simple(lambda v: float(v @ v),
    ...                       x0=np.array([5.0, -3.0]), sigma0=1.0,
    ...                       n_gen=50, seed=0)
    >>> np.linalg.norm(x) < 0.5
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def cma_es_simple(
    objective: Callable[[NDArray[np.float64]], float],
    x0: NDArray[np.float64], sigma0: float = 1.0,
    *, lam: int | None = None, mu: int | None = None,
    n_gen: int = 100, seed: int | None = 0,
) -> tuple[NDArray[np.float64], float]:
    rng = np.random.default_rng(seed)
    n = x0.size
    lam = lam if lam is not None else 4 + int(3 * np.log(n))
    mu = mu if mu is not None else lam // 2
    mean = np.asarray(x0, dtype=np.float64).ravel().copy()
    # diagonal cov
    C_diag = np.ones(n)
    sigma = float(sigma0)
    best_x = mean.copy()
    best_f = float(objective(mean))

    # recomb weights
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()

    gen = 0
    while gen < n_gen:
        # sample λ offspring
        samples = mean + sigma * np.sqrt(C_diag) * rng.standard_normal((lam, n))
        fvals = np.fromiter(
            map(lambda s: float(objective(s)), samples),
            dtype=np.float64,
            count=lam,
        )
        order = np.argsort(fvals)
        best_slice = samples[order[:mu]]
        new_mean = w @ best_slice
        # update diagonal C via weighted variance
        diff = best_slice - mean
        new_C = w @ (diff ** 2) / max(sigma ** 2, 1e-30)
        # smooth
        C_diag = 0.5 * C_diag + 0.5 * np.maximum(new_C, 1e-12)
        mean = new_mean
        # step-size: simple — scale by ratio of selected mean / expected
        sigma = float(sigma * np.exp(0.1 * (np.linalg.norm(mean - best_slice.mean(axis=0)) / (sigma + 1e-30) - 1.0)))
        sigma = max(sigma, 1e-12)
        if fvals[order[0]] < best_f:
            best_f = float(fvals[order[0]])
            best_x = samples[order[0]].copy()
        gen += 1
    return best_x, best_f


__all__ = ["cma_es_simple"]
