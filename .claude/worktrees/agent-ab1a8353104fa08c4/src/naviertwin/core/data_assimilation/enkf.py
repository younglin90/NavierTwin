"""Ensemble Kalman Filter (EnKF) — 표준 stochastic 버전.

상태 차원 n, 관측 차원 m, 앙상블 크기 N.

Forecast → Analysis:
    x_f = f(x_a) + η_f       (모델 잡음)
    y_pert = y_obs + η_obs    (관측 잡음)
    K = Cov(x_f, Hx_f) · (Cov(Hx_f, Hx_f) + R)^{-1}
    x_a = x_f + K · (y_pert - H x_f)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.enkf import EnKF
    >>> rng = np.random.default_rng(0)
    >>> n, N = 5, 20
    >>> ens = rng.standard_normal((N, n))
    >>> H = np.eye(n)[:3]
    >>> R = 0.01 * np.eye(3)
    >>> kf = EnKF(H=H, R=R)
    >>> y = np.zeros(3)
    >>> ens_new = kf.analysis(ens, y)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")

logger = get_logger(__name__)


class EnKF:
    """Stochastic Ensemble Kalman Filter.

    Args:
        H: 관측 연산자 (m, n).
        R: 관측 오차 공분산 (m, m).
        inflation: 앙상블 분산 inflation factor (>=1).
    """

    def __init__(
        self,
        H: NDArray[np.float64],
        R: NDArray[np.float64],
        inflation: float = 1.0,
    ) -> None:
        H = np.asarray(H, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        if H.ndim != 2 or R.ndim != 2:
            raise ValueError("H, R 는 2D 이어야 합니다")
        if R.shape[0] != R.shape[1] or R.shape[0] != H.shape[0]:
            raise ValueError(
                f"R shape={R.shape} != (H rows={H.shape[0]})"
            )
        if inflation < 1.0:
            raise ValueError(f"inflation 는 >= 1 (값={inflation})")
        self.H = H
        self.R = R
        self.inflation = float(inflation)

    def analysis(
        self,
        ensemble: NDArray[np.float64],
        observation: NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """한 스텝 analysis 를 수행한다.

        Args:
            ensemble: (N, n) 앙상블.
            observation: (m,) 관측값.
            rng: numpy random Generator (재현성용). None 이면 기본 rng.

        Returns:
            갱신된 앙상블 (N, n).
        """
        ens = np.asarray(ensemble, dtype=np.float64)
        y = np.asarray(observation, dtype=np.float64).ravel()
        if ens.ndim != 2:
            raise ValueError(f"ensemble (N,n) 2D 필요: {ens.shape}")
        if y.size != self.H.shape[0]:
            raise ValueError(
                f"observation 크기({y.size}) != H rows({self.H.shape[0]})"
            )
        if rng is None:
            rng = np.random.default_rng()

        N = ens.shape[0]

        # Inflation
        if self.inflation != 1.0:
            mean = ens.mean(axis=0, keepdims=True)
            ens = mean + self.inflation * (ens - mean)

        # Forecast observation 앙상블
        HX = ens @ self.H.T  # (N, m)

        mean_ens = ens.mean(axis=0)
        mean_HX = HX.mean(axis=0)
        A = ens - mean_ens
        D = HX - mean_HX

        Pxy = (A.T @ D) / (N - 1)  # (n, m)
        Pyy = (D.T @ D) / (N - 1) + self.R  # (m, m)

        K = np.asarray(_kernels.solve_square(Pyy.T, Pxy.T), dtype=np.float64).T  # (n, m)

        # 관측 교란
        noise = rng.multivariate_normal(
            mean=np.zeros(y.size), cov=self.R, size=N
        )
        innov = y[None, :] + noise - HX  # (N, m)

        updated = ens + innov @ K.T
        logger.debug(
            "EnKF analysis: N=%d, trace(Pyy)=%.4g", N, float(np.trace(Pyy))
        )
        return updated


__all__ = ["EnKF"]
