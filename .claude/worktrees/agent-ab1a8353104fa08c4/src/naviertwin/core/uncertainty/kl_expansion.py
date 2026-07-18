"""Karhunen-Loève expansion — covariance eigen-decomposition.

GP(0, C) ≈ Σ √λ_k φ_k(x) ξ_k,  ξ_k ~ N(0, 1).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.kl_expansion import (
    ...     kl_decompose, kl_sample,
    ... )
    >>> x = np.linspace(0, 1, 50)
    >>> def cov(x, y): return np.exp(-((x[:, None] - y[None, :])**2) / 0.1)
    >>> evals, evecs = kl_decompose(cov(x, x), n_modes=5)
    >>> sample = kl_sample(evals, evecs, rng=np.random.default_rng(0))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def kl_decompose(
    cov_matrix: NDArray[np.float64], n_modes: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """eigen-decompose C → (λ_k, φ_k), 내림차순."""
    C = np.asarray(cov_matrix, dtype=np.float64)
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    order = np.argsort(w)[::-1]
    w = w[order][:n_modes]
    V = V[:, order][:, :n_modes]
    w = np.maximum(w, 0)
    return w, V


def kl_sample(
    evals: NDArray[np.float64],
    evecs: NDArray[np.float64],
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    rng = rng if rng is not None else np.random.default_rng(0)
    xi = rng.standard_normal(evals.shape[0])
    return evecs @ (np.sqrt(evals) * xi)


__all__ = ["kl_decompose", "kl_sample"]
