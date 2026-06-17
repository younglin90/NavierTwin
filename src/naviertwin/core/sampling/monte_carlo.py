"""Monte Carlo 적분 + 중요도 샘플링 + antithetic.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.sampling.monte_carlo import mc_integral
    >>> est, err = mc_integral(lambda x: x**2, low=0, high=1, n=100000, seed=0)
    >>> abs(est - 1/3) < 0.01
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def mc_integral(
    f: Callable[[NDArray], NDArray | float],
    low: float, high: float,
    *, n: int = 10000, seed: int | None = 0,
) -> tuple[float, float]:
    """1D MC ∫_low^high f(x) dx. 반환: (추정, std err)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(low, high, n)
    y = np.asarray(f(x), dtype=np.float64)
    mean = float(y.mean())
    se = float(y.std(ddof=1) / np.sqrt(n))
    vol = high - low
    return vol * mean, vol * se


def mc_integral_antithetic(
    f: Callable[[NDArray], NDArray | float],
    low: float, high: float,
    *, n: int = 10000, seed: int | None = 0,
) -> tuple[float, float]:
    """Antithetic: x 와 low+high-x 의 평균."""
    rng = np.random.default_rng(seed)
    half = n // 2
    x = rng.uniform(low, high, half)
    x_pair = low + high - x
    y = 0.5 * (np.asarray(f(x)) + np.asarray(f(x_pair)))
    mean = float(y.mean())
    se = float(y.std(ddof=1) / np.sqrt(half))
    vol = high - low
    return vol * mean, vol * se


def mc_multivariate(
    f: Callable[[NDArray], NDArray | float],
    bounds: list[tuple[float, float]],
    *, n: int = 10000, seed: int | None = 0,
) -> tuple[float, float]:
    """다차원 박스 MC 적분."""
    rng = np.random.default_rng(seed)
    d = len(bounds)
    X = np.zeros((n, d))
    vol = 1.0
    i = 0
    while i < d:
        a, b = bounds[i]
        X[:, i] = rng.uniform(a, b, n)
        vol *= (b - a)
        i += 1
    y = np.asarray(f(X), dtype=np.float64)
    mean = float(y.mean())
    se = float(y.std(ddof=1) / np.sqrt(n))
    return vol * mean, vol * se


def importance_sample(
    f: Callable[[NDArray], NDArray | float],
    q_sample: Callable[[int, np.random.Generator], NDArray],
    p_pdf: Callable[[NDArray], NDArray | float],
    q_pdf: Callable[[NDArray], NDArray | float],
    *, n: int = 10000, seed: int | None = 0,
) -> tuple[float, float]:
    """E_p[f] ≈ (1/n) Σ f(x) p(x)/q(x), x~q."""
    rng = np.random.default_rng(seed)
    x = q_sample(n, rng)
    w = np.asarray(p_pdf(x)) / (np.asarray(q_pdf(x)) + 1e-30)
    y = np.asarray(f(x)) * w
    mean = float(y.mean())
    se = float(y.std(ddof=1) / np.sqrt(n))
    return mean, se


__all__ = [
    "mc_integral", "mc_integral_antithetic", "mc_multivariate",
    "importance_sample",
]
