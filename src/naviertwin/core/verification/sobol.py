"""Sobol sensitivity — Saltelli matrix-style first-order + total.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.verification.sobol import sobol_indices
    >>> rng = np.random.default_rng(0)
    >>> def model(X): return X[:, 0] + 0.1 * X[:, 1]
    >>> S, ST = sobol_indices(model, n_dim=2, n_samples=2000, rng=rng)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def sobol_indices(
    model: Callable[[NDArray], NDArray], *,
    n_dim: int, n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, NDArray]:
    """Returns (S_i, S_T_i)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    A = rng.uniform(0, 1, (n_samples, n_dim))
    B = rng.uniform(0, 1, (n_samples, n_dim))
    fA = model(A)
    fB = model(B)
    var_y = float(np.var(np.concatenate([fA, fB])) + 1e-30)
    AB = np.repeat(A[np.newaxis, :, :], n_dim, axis=0)
    idx = np.arange(n_dim)
    AB[idx, :, idx] = B.T
    fAB = model(AB.reshape(n_dim * n_samples, n_dim)).reshape(n_dim, n_samples)
    S = np.mean(fB[np.newaxis, :] * (fAB - fA[np.newaxis, :]), axis=1) / var_y
    ST = 0.5 * np.mean((fA[np.newaxis, :] - fAB) ** 2, axis=1) / var_y
    return S, ST


__all__ = ["sobol_indices"]
