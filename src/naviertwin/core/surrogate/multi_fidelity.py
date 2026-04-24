"""Multi-fidelity linear surrogate — Kennedy-O'Hagan-lite:
y_hi(x) = ρ · y_lo(x) + δ(x),  ρ는 스칼라, δ는 보정항.

간단화: lo/hi surrogate 는 scratch Kriging 재사용.

Examples:
    >>> # test 참조
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.surrogate.kriging_scratch import OrdinaryKriging


class LinearMultiFidelity:
    def __init__(self, theta: float = 0.5, nugget: float = 1e-6) -> None:
        self.theta = float(theta)
        self.nugget = float(nugget)
        self.lo: OrdinaryKriging | None = None
        self.delta: OrdinaryKriging | None = None
        self.rho: float = 1.0

    def fit(
        self, X_lo: NDArray[np.float64], y_lo: NDArray[np.float64],
        X_hi: NDArray[np.float64], y_hi: NDArray[np.float64],
    ) -> "LinearMultiFidelity":
        self.lo = OrdinaryKriging(self.theta, self.nugget).fit(X_lo, y_lo)
        # evaluate lo at X_hi
        y_lo_at_hi = self.lo.predict(X_hi)
        # ρ 추정: y_hi = ρ y_lo_at_hi + δ → least squares ρ
        y_hi_r = np.asarray(y_hi).ravel()
        self.rho = float((y_lo_at_hi @ y_hi_r) / (y_lo_at_hi @ y_lo_at_hi + 1e-30))
        resid = y_hi_r - self.rho * y_lo_at_hi
        self.delta = OrdinaryKriging(self.theta, self.nugget).fit(X_hi, resid)
        return self

    def predict(self, Xq: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.lo is None or self.delta is None:
            raise RuntimeError("fit 먼저")
        return self.rho * self.lo.predict(Xq) + self.delta.predict(Xq)


__all__ = ["LinearMultiFidelity"]
