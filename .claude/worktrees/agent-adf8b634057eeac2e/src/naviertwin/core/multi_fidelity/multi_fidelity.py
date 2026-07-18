"""멀티피델리티 surrogate — 간단한 additive co-Kriging.

    y_H(x) ≈ ρ · y_L(x) + δ(x)

저해상도 surrogate y_L 와 고해상도 잔차 δ 를 각각 Kriging 으로 모델링.
scikit-learn GaussianProcessRegressor 기반.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.multi_fidelity.multi_fidelity import AdditiveCoKriging
    >>> rng = np.random.default_rng(0)
    >>> X_L = rng.uniform(-1, 1, (30, 2))
    >>> X_H = X_L[:10]
    >>> y_L = np.sin(X_L[:, 0] + X_L[:, 1])
    >>> y_H = y_L[:10] + 0.1 * np.cos(X_H[:, 0])
    >>> mf = AdditiveCoKriging()
    >>> mf.fit(X_L, y_L, X_H, y_H)
    >>> mf.predict(np.array([[0.0, 0.0]])).shape
    (1,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class AdditiveCoKriging:
    """단순 additive Co-Kriging (ρ 를 최소제곱으로 추정)."""

    def __init__(self) -> None:
        self._gp_low: object | None = None
        self._gp_delta: object | None = None
        self.rho_: float = 1.0
        self.is_fitted: bool = False

    def fit(
        self,
        X_low: NDArray[np.float64],
        y_low: NDArray[np.float64],
        X_high: NDArray[np.float64],
        y_high: NDArray[np.float64],
    ) -> None:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        except ImportError as exc:
            raise RuntimeError("scikit-learn 필요") from exc

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self._gp_low = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self._gp_low.fit(X_low, y_low)

        # 고해상도 점에서 저해상도 예측
        y_low_at_high = self._gp_low.predict(X_high)
        # ρ 추정: y_high ≈ ρ · y_low_at_high + δ (δ 는 평균 0 가정 시 선형 최소제곱)
        denom = float(np.dot(y_low_at_high, y_low_at_high) + 1e-12)
        self.rho_ = float(np.dot(y_low_at_high, y_high) / denom)
        delta = y_high - self.rho_ * y_low_at_high

        self._gp_delta = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self._gp_delta.fit(X_high, delta)

        self.is_fitted = True
        logger.info(
            "AdditiveCoKriging fit: ρ=%.4f, n_low=%d, n_high=%d",
            self.rho_, len(X_low), len(X_high),
        )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출하세요")
        y_low = self._gp_low.predict(X)
        delta = self._gp_delta.predict(X)
        return self.rho_ * y_low + delta


__all__ = ["AdditiveCoKriging"]
