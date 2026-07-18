"""Metropolis-Hastings MCMC 샘플러.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.mcmc import metropolis_hastings
    >>> # log-density: N(0, 1)
    >>> logp = lambda x: -0.5 * x[0]**2
    >>> samples = metropolis_hastings(logp, x0=np.zeros(1), n=5000, step=0.5, seed=0)
    >>> abs(samples.mean()) < 0.2
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def metropolis_hastings(
    log_prob: Callable[[NDArray[np.float64]], float],
    x0: NDArray[np.float64],
    *, n: int = 10000, step: float = 0.5, burn: int = 500,
    seed: int | None = 0,
) -> NDArray[np.float64]:
    """isotropic Gaussian proposal MH."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    d = x.size
    out = np.zeros((n, d))
    lp = log_prob(x)
    accepts = 0
    i = 0
    while i < n + burn:
        prop = x + rng.normal(0, step, size=d)
        lp_new = log_prob(prop)
        if np.log(rng.random()) < lp_new - lp:
            x = prop
            lp = lp_new
            accepts += 1
        if i >= burn:
            out[i - burn] = x
        i += 1
    # 사용자 접근을 위해 acceptance rate 을 attribute 가 아닌 return 으로 숨겨 둠 (단순화)
    return out


def acceptance_rate(
    log_prob: Callable[[NDArray[np.float64]], float],
    x0: NDArray[np.float64], *, n: int = 1000, step: float = 0.5,
    seed: int | None = 0,
) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    d = x.size
    lp = log_prob(x)
    acc = 0
    i = 0
    while i < n:
        prop = x + rng.normal(0, step, size=d)
        lp_new = log_prob(prop)
        if np.log(rng.random()) < lp_new - lp:
            x = prop
            lp = lp_new
            acc += 1
        i += 1
    return acc / n


__all__ = ["metropolis_hastings", "acceptance_rate"]
