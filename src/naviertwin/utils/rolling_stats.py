"""Welford 알고리즘 기반 온라인 평균/분산 — streaming twin 용.

Examples:
    >>> from naviertwin.utils.rolling_stats import WelfordStats
    >>> s = WelfordStats()
    >>> s.update(1.0); s.update(2.0); s.update(3.0); s.update(4.0)
    >>> round(s.mean, 3), round(s.variance, 3)
    (2.5, 1.667)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class WelfordStats:
    """스칼라 또는 벡터에 대한 온라인 평균/분산."""

    def __init__(self) -> None:
        self.n: int = 0
        self._mean: NDArray[np.float64] | float = 0.0
        self._m2: NDArray[np.float64] | float = 0.0

    def update(self, x: float | NDArray[np.float64]) -> None:
        xa = np.asarray(x, dtype=np.float64)
        if self.n == 0:
            self._mean = np.zeros_like(xa) if xa.ndim > 0 else 0.0
            self._m2 = np.zeros_like(xa) if xa.ndim > 0 else 0.0
        self.n += 1
        delta = xa - self._mean
        self._mean = self._mean + delta / self.n
        delta2 = xa - self._mean
        self._m2 = self._m2 + delta * delta2

    @property
    def mean(self) -> float | NDArray[np.float64]:
        if isinstance(self._mean, np.ndarray):
            return self._mean.copy()
        return float(self._mean)

    @property
    def variance(self) -> float | NDArray[np.float64]:
        if self.n < 2:
            return 0.0 if not isinstance(self._m2, np.ndarray) else np.zeros_like(self._m2)
        v = self._m2 / (self.n - 1)
        return float(v) if not isinstance(v, np.ndarray) else v

    @property
    def std(self) -> float | NDArray[np.float64]:
        v = self.variance
        return float(np.sqrt(v)) if not isinstance(v, np.ndarray) else np.sqrt(v)

    def merge(self, other: "WelfordStats") -> "WelfordStats":
        """두 stats 병합 (parallel reduction 지원)."""
        if self.n == 0:
            self.n = other.n
            self._mean = other._mean
            self._m2 = other._m2
            return self
        if other.n == 0:
            return self
        n = self.n + other.n
        delta = other._mean - self._mean  # type: ignore[operator]
        new_mean = self._mean + delta * (other.n / n)
        new_m2 = self._m2 + other._m2 + delta * delta * (self.n * other.n / n)
        self.n = n
        self._mean = new_mean
        self._m2 = new_m2
        return self


__all__ = ["WelfordStats"]
