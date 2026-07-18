"""스냅샷 정규화 — MinMax / Standard(z-score) / Robust(median/IQR) / MaxAbs.

ROM/neural surrogate 학습 전 필수. fit → transform → inverse_transform 패턴.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.preprocessing.normalizer import Normalizer
    >>> n = Normalizer("standard")
    >>> Y = n.fit_transform(np.array([[1., 2.], [3., 4.]]))
    >>> np.allclose(n.inverse_transform(Y), [[1., 2.], [3., 4.]])
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Normalizer:
    """축-정렬 정규화. axis=0 기준 컬럼별 통계를 학습."""

    _KINDS = frozenset({"minmax", "standard", "robust", "maxabs"})

    def __init__(self, kind: str = "standard", *, axis: int = 0) -> None:
        if kind not in self._KINDS:
            raise ValueError(f"kind ∈ {self._KINDS}, got {kind}")
        self.kind = kind
        self.axis = int(axis)
        self.center_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None
        self.is_fitted: bool = False

    def fit(self, X: NDArray[np.float64]) -> "Normalizer":
        X = np.asarray(X, dtype=np.float64)
        if self.kind == "minmax":
            self.center_ = X.min(axis=self.axis, keepdims=True)
            rng = X.max(axis=self.axis, keepdims=True) - self.center_
            self.scale_ = np.where(rng == 0, 1.0, rng)
        elif self.kind == "standard":
            self.center_ = X.mean(axis=self.axis, keepdims=True)
            sd = X.std(axis=self.axis, keepdims=True)
            self.scale_ = np.where(sd == 0, 1.0, sd)
        elif self.kind == "robust":
            self.center_ = np.median(X, axis=self.axis, keepdims=True)
            q75 = np.quantile(X, 0.75, axis=self.axis, keepdims=True)
            q25 = np.quantile(X, 0.25, axis=self.axis, keepdims=True)
            iqr = q75 - q25
            self.scale_ = np.where(iqr == 0, 1.0, iqr)
        elif self.kind == "maxabs":
            self.center_ = np.zeros_like(X.mean(axis=self.axis, keepdims=True))
            mx = np.max(np.abs(X), axis=self.axis, keepdims=True)
            self.scale_ = np.where(mx == 0, 1.0, mx)
        self.is_fitted = True
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        self._check()
        return (np.asarray(X, dtype=np.float64) - self.center_) / self.scale_

    def inverse_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        self._check()
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.center_

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.fit(X).transform(X)

    def _check(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출하세요")


__all__ = ["Normalizer"]
