"""Quasi-Monte Carlo 샘플러 — Halton, Sobol (via scipy), Latin Hypercube.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.sampling.qmc import halton, latin_hypercube
    >>> X = halton(n=64, d=3)
    >>> X.shape
    (64, 3)
    >>> L = latin_hypercube(n=32, d=2, seed=0)
    >>> L.shape
    (32, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _halton_1d(n: int, base: int) -> NDArray[np.float64]:
    """1D Halton sequence — radical inverse in given base."""
    out = np.zeros(n)
    i = 0
    while i < n:
        f = 1.0
        r = 0.0
        k = i + 1
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        out[i] = r
        i += 1
    return out


def halton(n: int, d: int) -> NDArray[np.float64]:
    """d 차원 Halton 샘플 (n, d) — base 는 처음 d개 소수."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    if d > len(primes):
        raise ValueError(f"d <= {len(primes)} 필요")
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by halton")
    return _kernels.halton_sequence(n, d)


def latin_hypercube(
    n: int, d: int, seed: int | None = None
) -> NDArray[np.float64]:
    """Latin Hypercube Sampling — 각 차원을 n 개 구간으로 분할."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, d))
    j = 0
    while j < d:
        perm = rng.permutation(n)
        u = rng.random(n)
        out[:, j] = (perm + u) / n
        j += 1
    return out


def sobol(n: int, d: int, seed: int | None = None) -> NDArray[np.float64]:
    """scipy.stats.qmc.Sobol 래퍼 — 없으면 halton 으로 폴백."""
    try:
        from scipy.stats import qmc

        engine = qmc.Sobol(d=d, scramble=True, seed=seed)
        return np.asarray(engine.random(n), dtype=np.float64)
    except Exception:
        logger.warning("scipy Sobol 실패 → Halton 폴백")
        return halton(n, d)


def scale_to_bounds(
    unit_samples: NDArray[np.float64],
    bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """[0, 1]^d 샘플을 bounds 로 스케일."""
    unit = np.asarray(unit_samples, dtype=np.float64)
    b = np.asarray(bounds, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 2 or b.shape[0] != unit.shape[1]:
        raise ValueError(
            f"bounds shape {b.shape} != (d={unit.shape[1]}, 2)"
        )
    return b[:, 0] + unit * (b[:, 1] - b[:, 0])


__all__ = ["halton", "latin_hypercube", "sobol", "scale_to_bounds"]
