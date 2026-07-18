"""Conformal prediction — distribution-free coverage 보장 구간.

split conformal: 보정 집합의 residual quantile 로 예측 구간 폭 결정.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.conformal import SplitConformal
    >>> cp = SplitConformal(alpha=0.1)
    >>> cp.calibrate(y_true=np.zeros(100), y_pred=0.1 * np.random.default_rng(0).standard_normal(100))
    >>> cp.q > 0
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SplitConformal:
    def __init__(self, alpha: float = 0.1) -> None:
        if not (0 < alpha < 1):
            raise ValueError("alpha ∈ (0, 1)")
        self.alpha = float(alpha)
        self.q: float | None = None

    def calibrate(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64],
    ) -> "SplitConformal":
        r = np.abs(np.asarray(y_true, dtype=np.float64) -
                   np.asarray(y_pred, dtype=np.float64))
        n = r.size
        # finite-sample corrected quantile
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q = float(np.quantile(r, level))
        return self

    def predict_interval(
        self, y_pred: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.q is None:
            raise RuntimeError("calibrate 먼저")
        y = np.asarray(y_pred, dtype=np.float64)
        return y - self.q, y + self.q

    def coverage(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64],
    ) -> float:
        lo, hi = self.predict_interval(y_pred)
        return float(np.mean((y_true >= lo) & (y_true <= hi)))


__all__ = ["SplitConformal"]
