"""Latent dynamics linearization — fit linear A from {z_k → z_{k+1}}.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.latent_linearize import (
    ...     fit_latent_A,
    ... )
    >>> Z = np.random.default_rng(0).standard_normal((20, 3))
    >>> A = fit_latent_A(Z)
    >>> A.shape
    (3, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fit_latent_A(Z: NDArray[np.float64]) -> NDArray[np.float64]:
    """least-squares: Z2 = A Z1.  Z shape (T, r)."""
    Z = np.asarray(Z, dtype=np.float64)
    Z1 = Z[:-1].T  # (r, T-1)
    Z2 = Z[1:].T
    A, *_ = np.linalg.lstsq(Z1.T, Z2.T, rcond=None)
    return A.T  # (r, r)


def predict_latent(A: NDArray, z0: NDArray, n_steps: int) -> NDArray:
    z = z0.copy()
    out = np.empty((n_steps + 1, z.size), dtype=np.asarray(z).dtype)
    out[0] = z
    step = 0
    while step < n_steps:
        z = A @ z
        out[step + 1] = z
        step += 1
    return out


__all__ = ["fit_latent_A", "predict_latent"]
