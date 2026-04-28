"""실시간 누적 통계 모먼트 (Welford / Pébay).

대용량 CFD 시계열에서 메모리 한 줄로 평균/분산/공분산을 점증 계산.
긴 LES/DNS 데이터의 in-situ 통계 추출 표준.

상용 툴 대응:
    - Ansys Fluent: Sample for Time Statistics (running)
    - Tecplot 360: Time-Average (rolling)
    - EnSight: Variable Statistics

References:
    Welford, B.P., "Note on a method for calculating corrected sums of
    squares and products", Technometrics, 1962.
    Pébay, P., "Formulas for Robust, One-Pass Parallel Computation of
    Covariances and Arbitrary-Order Statistical Moments", SAND, 2008.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> from naviertwin.core.flow_analysis.running_moments import RunningMoments
    >>> rm = RunningMoments(shape=(3,))
    >>> for _ in range(100):
    ...     rm.update(rng.standard_normal(3))
    >>> abs(rm.mean[0]) < 0.5
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class RunningMoments:
    """Welford 누적 평균/분산 (1차+2차 모먼트).

    Attributes:
        shape: 변수 형상 (예: (n_space,) 또는 (n_y, n_x)).
        n: 처리된 샘플 수.
        mean: 누적 평균.
        M2: ⟨(x - mean)²⟩ 의 누적 합.
    """

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.shape = tuple(shape)
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)

    def update(self, x: NDArray[np.float64] | float) -> None:
        """단일 샘플 갱신."""
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.shape != self.shape:
            raise ValueError(
                f"sample shape {x_arr.shape} != initialized {self.shape}"
            )
        self.n += 1
        delta = x_arr - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = x_arr - self.mean
        self.M2 = self.M2 + delta * delta2

    def merge(self, other: "RunningMoments") -> None:
        """다른 RunningMoments 병합 (병렬 합산용).

        Pébay 2008 공식.
        """
        if other.shape != self.shape:
            raise ValueError(
                f"shape mismatch: {self.shape} vs {other.shape}"
            )
        if other.n == 0:
            return
        if self.n == 0:
            self.n = other.n
            self.mean = other.mean.copy()
            self.M2 = other.M2.copy()
            return
        n_a = self.n
        n_b = other.n
        n_ab = n_a + n_b
        delta = other.mean - self.mean
        self.mean = self.mean + delta * n_b / n_ab
        self.M2 = self.M2 + other.M2 + delta ** 2 * n_a * n_b / n_ab
        self.n = n_ab

    @property
    def variance(self) -> NDArray[np.float64]:
        """편향 분산 σ² (Welford에서는 M2/n)."""
        if self.n < 1:
            return np.zeros(self.shape)
        return self.M2 / self.n

    @property
    def variance_unbiased(self) -> NDArray[np.float64]:
        """비편향 표본 분산 σ² (M2 / (n-1))."""
        if self.n < 2:
            return np.zeros(self.shape)
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> NDArray[np.float64]:
        """표준편차."""
        return np.sqrt(self.variance)

    def reset(self) -> None:
        """누적 상태 초기화."""
        self.n = 0
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.M2 = np.zeros(self.shape, dtype=np.float64)


class RunningCovariance:
    """두 변수의 누적 공분산 ⟨(x - x̄)(y - ȳ)⟩."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.shape = tuple(shape)
        self.n = 0
        self.mean_x = np.zeros(shape, dtype=np.float64)
        self.mean_y = np.zeros(shape, dtype=np.float64)
        self.C = np.zeros(shape, dtype=np.float64)  # co-moment
        self.M2_x = np.zeros(shape, dtype=np.float64)
        self.M2_y = np.zeros(shape, dtype=np.float64)

    def update(
        self,
        x: NDArray[np.float64] | float,
        y: NDArray[np.float64] | float,
    ) -> None:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if x_arr.shape != self.shape or y_arr.shape != self.shape:
            raise ValueError(
                f"sample shape mismatch: {x_arr.shape}/{y_arr.shape} vs {self.shape}"
            )
        self.n += 1
        dx = x_arr - self.mean_x
        self.mean_x = self.mean_x + dx / self.n
        dy = y_arr - self.mean_y
        self.mean_y = self.mean_y + dy / self.n
        # cross moment
        self.C = self.C + dx * (y_arr - self.mean_y)
        # individual moments
        self.M2_x = self.M2_x + dx * (x_arr - self.mean_x)
        self.M2_y = self.M2_y + dy * (y_arr - self.mean_y)

    @property
    def covariance(self) -> NDArray[np.float64]:
        """⟨(x - x̄)(y - ȳ)⟩ (편향 추정)."""
        if self.n < 1:
            return np.zeros(self.shape)
        return self.C / self.n

    @property
    def correlation(self) -> NDArray[np.float64]:
        """Pearson 상관 계수 ∈ [-1, 1]."""
        if self.n < 2:
            return np.zeros(self.shape)
        sx = np.sqrt(self.M2_x / self.n)
        sy = np.sqrt(self.M2_y / self.n)
        denom = np.maximum(sx * sy, 1e-30)
        return (self.C / self.n) / denom


def online_mean_var(
    x_stream: NDArray[np.float64],
) -> tuple[float, float]:
    """단순 1D 스트림에서 Welford 평균/분산 반환 (one-pass).

    Args:
        x_stream: (N,) 또는 iterable of floats.

    Returns:
        (mean, variance_unbiased).
    """
    arr = np.asarray(x_stream, dtype=np.float64).ravel()
    rm = RunningMoments(shape=())
    for v in arr:
        rm.update(float(v))
    return float(rm.mean), float(rm.variance_unbiased)


__all__ = ["RunningMoments", "RunningCovariance", "online_mean_var"]
