"""Particle smoother — backward weight reweighting (Doucet et al.).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.particle_smoother import smooth_particles
    >>> rng = np.random.default_rng(0)
    >>> particles = rng.standard_normal((5, 100))
    >>> weights = np.full((5, 100), 1/100)
    >>> ws = smooth_particles(particles, weights, lambda x_to, x_from: 1.0)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def smooth_particles(
    particles: NDArray[np.float64],
    weights: NDArray[np.float64],
    transition_density: Callable[[float, float], float],
) -> NDArray[np.float64]:
    """Backward weight: w_s[k, i] = w[k, i] Σ_j w_s[k+1, j] p(x_{k+1, j} | x_{k, i}) / Σ ..."""
    p = np.asarray(particles, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).copy()
    N, M = p.shape
    w_s = w.copy()
    w_s[-1] = w[-1]
    for k in range(N - 2, -1, -1):
        # for each particle i at time k
        for i in range(M):
            num = 0.0
            for j in range(M):
                tij = transition_density(float(p[k + 1, j]), float(p[k, i]))
                # marginal denom (sum_m w[k, m] p(j|m))
                d = sum(
                    w[k, m] * transition_density(float(p[k + 1, j]), float(p[k, m]))
                    for m in range(M)
                )
                if d > 0:
                    num += w_s[k + 1, j] * tij / d
            w_s[k, i] = w[k, i] * num
        # normalize
        s = w_s[k].sum()
        if s > 0:
            w_s[k] /= s
    return w_s


__all__ = ["smooth_particles"]
