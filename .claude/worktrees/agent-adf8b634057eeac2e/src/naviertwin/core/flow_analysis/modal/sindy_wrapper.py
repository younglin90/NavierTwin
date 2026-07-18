"""SINDy (Sparse Identification of Nonlinear Dynamics) 래퍼.

시계열 x(t) 에서 dx/dt 를 유한차분으로 추정하고, 후보 함수 라이브러리
Θ(x) 와 STLSQ (sequentially-thresholded least squares) 로 희소 계수를 찾는다.

PySINDy 설치 시 그 백엔드를 사용하고, 미설치 시 자체 구현으로 폴백.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_analysis.modal.sindy_wrapper import SINDy
    >>> # Lorenz-like 2D 시뮬레이션 데이터
    >>> t = np.linspace(0, 5, 500)
    >>> dt = t[1] - t[0]
    >>> x = np.sin(t)
    >>> y = np.cos(t)
    >>> X = np.column_stack([x, y])
    >>> s = SINDy(poly_degree=2, threshold=0.1)
    >>> s.fit(X, dt=dt)
    >>> s.coef_.shape
    (2, 6)
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _combo_name(combo: tuple[int, ...]) -> str:
    parts: list[str] = []
    idx = 0
    while idx < len(combo):
        parts.append(f"x{combo[idx]}")
        idx += 1
    return "*".join(parts)


def _polynomial_library(
    X: NDArray[np.float64], degree: int
) -> tuple[NDArray[np.float64], list[str]]:
    """X (N, n) → Θ (N, p), feature names."""
    X = np.asarray(X, dtype=np.float64)
    N, n = X.shape
    combos: list[tuple[int, ...]] = []
    d = 1
    while d <= degree:
        combos.extend(combinations_with_replacement(range(n), d))
        d += 1
    if combos:
        columns = tuple(map(lambda combo: np.prod(X[:, combo], axis=1), combos))
        theta = np.column_stack((np.ones(N), *columns))
    else:
        theta = np.ones((N, 1), dtype=np.float64)
    return theta, ["1", *tuple(map(_combo_name, combos))]


def _stlsq(
    Theta: NDArray[np.float64],
    dX: NDArray[np.float64],
    threshold: float,
    max_iter: int = 10,
) -> NDArray[np.float64]:
    """Sequentially-thresholded LSQ solving Ξ in Theta Ξ ≈ dX."""
    Xi, *_ = np.linalg.lstsq(Theta, dX, rcond=None)
    it = 0
    while it < max_iter:
        small = np.abs(Xi) < threshold
        Xi[small] = 0
        j = 0
        while j < Xi.shape[1]:
            big = ~small[:, j]
            if np.any(big):
                Xi[big, j], *_ = np.linalg.lstsq(
                    Theta[:, big], dX[:, j], rcond=None
                )
            j += 1
        it += 1
    return Xi


class SINDy:
    """경량 SINDy — PySINDy 없이 동작."""

    def __init__(
        self,
        poly_degree: int = 2,
        threshold: float = 0.05,
    ) -> None:
        self.poly_degree = poly_degree
        self.threshold = threshold

        self.coef_: NDArray[np.float64] | None = None
        self.feature_names_: list[str] = []
        self.is_fitted: bool = False
        self._use_pysindy: bool = False
        self._pysindy_model: object | None = None

    def fit(self, X: NDArray[np.float64], dt: float) -> None:
        """시계열 X (N, n) 에서 dx/dt 를 추정하고 희소 회귀.

        Args:
            X: (N, n) 시계열.
            dt: 시간 간격.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or dt <= 0:
            raise ValueError(f"X 2D + dt>0 필요: X.shape={X.shape}, dt={dt}")

        # PySINDy 경로 — 있으면 그대로 사용
        try:
            import pysindy as ps

            model = ps.SINDy(
                optimizer=ps.STLSQ(threshold=self.threshold),
                feature_library=ps.PolynomialLibrary(degree=self.poly_degree),
            )
            model.fit(X, t=dt)
            self._pysindy_model = model
            self._use_pysindy = True
            # PySINDy coef_: shape = (n_targets, n_features)
            self.coef_ = np.asarray(model.coefficients(), dtype=np.float64)
            try:
                self.feature_names_ = list(model.get_feature_names())
            except Exception:  # noqa: BLE001
                self.feature_names_ = []
            self.is_fitted = True
            logger.info("SINDy(PySINDy) 학습 완료: coef=%s", self.coef_.shape)
            return
        except ImportError:
            pass

        # 자체 구현
        # 중심 차분으로 dx/dt 추정 (끝점은 forward/backward)
        dX = np.zeros_like(X)
        dX[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
        dX[0] = (X[1] - X[0]) / dt
        dX[-1] = (X[-1] - X[-2]) / dt

        Theta, names = _polynomial_library(X, self.poly_degree)
        Xi = _stlsq(Theta, dX, self.threshold)

        self.coef_ = Xi.T  # (n_targets, n_features)
        self.feature_names_ = names
        self.is_fitted = True
        logger.info("SINDy(내장) 학습 완료: coef shape=%s", self.coef_.shape)

    def equations(self, precision: int = 4) -> list[str]:
        """학습된 dx_i/dt = Σ c_ij · f_j(x) 문자열 리스트."""
        if not self.is_fitted or self.coef_ is None:
            raise RuntimeError("fit() 먼저 호출하세요")
        if self._use_pysindy and self._pysindy_model is not None:
            try:
                return list(self._pysindy_model.equations())
            except Exception:  # noqa: BLE001
                pass
        eqs: list[str] = []
        i = 0
        while i < self.coef_.shape[0]:
            row = self.coef_[i]
            terms: list[str] = []
            j = 0
            while j < row.shape[0]:
                c = row[j]
                if abs(c) > 0:
                    terms.append(f"{c:.{precision}g}*{self.feature_names_[j]}")
                j += 1
            eqs.append(f"dx{i}/dt = " + (" + ".join(terms) if terms else "0"))
            i += 1
        return eqs


__all__ = ["SINDy"]
