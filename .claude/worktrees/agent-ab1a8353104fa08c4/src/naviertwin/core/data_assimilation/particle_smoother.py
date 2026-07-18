"""Particle smoother - backward weight reweighting (Doucet et al.).

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
    transition_grid = np.frompyfunc(transition_density, 2, 1)
    k = N - 2
    while k >= 0:
        kernel = transition_grid(p[k + 1, :, None], p[k, None, :]).astype(np.float64)
        denom = kernel @ w[k]
        ratio = np.divide(w_s[k + 1], denom, out=np.zeros(M), where=denom > 0)
        w_s[k] = w[k] * (kernel.T @ ratio)
        s = w_s[k].sum()
        if s > 0:
            w_s[k] /= s
        k -= 1
    return w_s


__all__ = ["smooth_particles"]
