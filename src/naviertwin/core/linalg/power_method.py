"""Native power iteration, inverse iteration, and Rayleigh quotient."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def power_iteration(
    A: NDArray[np.float64], n_iter: int = 200, *,
    x0: NDArray | None = None, tol: float = 1e-10, seed: int | None = 0,
) -> tuple[float, NDArray[np.float64]]:
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) if x0 is None else np.asarray(x0).ravel()
    lam, vec = _kernels.power_iteration(A, int(n_iter), np.asarray(x, dtype=np.float64), float(tol))
    return float(lam), np.asarray(vec, dtype=np.float64)


def inverse_power(
    A: NDArray[np.float64], shift: float = 0.0, n_iter: int = 100,
    *, seed: int | None = 0,
) -> tuple[float, NDArray[np.float64]]:
    """(A - shift I)⁻¹ 에 대한 power iteration → shift 에 가까운 고유값."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    lam, vec = _kernels.inverse_power(A, float(shift), int(n_iter), np.asarray(x, dtype=np.float64))
    return float(lam), np.asarray(vec, dtype=np.float64)


def rayleigh_quotient(A: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    return float(_kernels.rayleigh_quotient(np.asarray(A, dtype=np.float64), np.asarray(x, dtype=np.float64).ravel()))


__all__ = ["power_iteration", "inverse_power", "rayleigh_quotient"]
