"""Sequential Monte Carlo — temperature ladder + resample.

p_t(x) ∝ prior · likelihood^β_t,  β: 0 → 1.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.smc import smc_sample
    >>> rng = np.random.default_rng(0)
    >>> def loglike(x): return -0.5 * x**2
    >>> samples = smc_sample(loglike, n_particles=200, n_steps=5, rng=rng)
    >>> samples.shape
    (200,)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def smc_sample(
    loglike: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    n_particles: int = 500,
    n_steps: int = 10,
    prior_std: float = 5.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """1D SMC: prior N(0, prior_std²) → posterior."""
    rng = rng if rng is not None else np.random.default_rng(0)
    x = rng.normal(0, prior_std, n_particles)
    betas = np.linspace(0.0, 1.0, n_steps + 1)
    for k in range(n_steps):
        dbeta = betas[k + 1] - betas[k]
        log_w = dbeta * loglike(x)
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        # resample
        idx = rng.choice(n_particles, n_particles, p=w)
        x = x[idx]
        # MH move
        prop = x + rng.normal(0, 0.5, n_particles)
        log_acc = betas[k + 1] * (loglike(prop) - loglike(x))
        accept = np.log(rng.uniform(size=n_particles)) < log_acc
        x = np.where(accept, prop, x)
    return x


__all__ = ["smc_sample"]
