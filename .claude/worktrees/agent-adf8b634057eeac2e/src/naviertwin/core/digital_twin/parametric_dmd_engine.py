"""ParametricDMD 트윈 엔진 — 비정상 파라미터 스윕의 (μ, t) 예보 (v5.2).

PyDMD 의 :class:`pydmd.ParametricDMD` (arXiv:2110.09155) 를 앱의 duck-typed
엔진 계약(``predict(params)``)으로 감싼다. 케이스별 시계열(μ마다 하나)을
학습하면, **학습에 없던 μ 에서의 시간 전개**를 보간하고 **학습 구간 밖 t**
까지 예보한다 — 단일 케이스 DMD(:class:`DMDTwinEngine`)의 파라메트릭 확장.

*partitioned* 방식(μ 케이스마다 DMD 하나)을 쓴다 — 케이스마다 고유 주파수가
달라도(예: 셔딩 주파수가 유속에 비례) 각자 적합하므로 monolithic 보다 안전하다.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["ParametricDMDTwinEngine"]


class ParametricDMDTwinEngine:
    """(μ₁..μₖ, t) → 필드 예보 엔진 — ParametricDMD 래퍼.

    Attributes:
        training_metadata: 앱 계약 메타데이터 (param_names/mins/maxs 등).
    """

    def __init__(
        self,
        pdmd: Any,
        *,
        time_steps: Sequence[float],
        forecast_factor: float = 1.5,
    ) -> None:
        """이미 ``fit`` 된 ParametricDMD 를 감싼다.

        Args:
            pdmd: 학습 완료된 :class:`pydmd.ParametricDMD`.
            time_steps: 학습 시계열의 물리 시간 (균일 간격 가정 — DMD 전제).
            forecast_factor: 예보 상한 = 학습 구간 × 이 배수.
        """
        times = [float(t) for t in time_steps]
        if len(times) < 2:
            raise ValueError("ParametricDMD 에는 타임스텝이 2개 이상 필요합니다.")
        self._pdmd = pdmd
        self._t0 = times[0]
        self._dt = float(times[1] - times[0])
        if self._dt <= 0:
            raise ValueError("time_steps 는 증가 수열이어야 합니다.")
        self._n_train_steps = len(times)
        self.forecast_factor = float(forecast_factor)

        # DMD 내부 시간(인덱스 공간)을 예보 구간까지 연장한다.
        n_forecast = max(
            self._n_train_steps,
            int(round((self._n_train_steps - 1) * self.forecast_factor)) + 1,
        )
        try:
            self._pdmd.dmd_time["tend"] = n_forecast - 1
        except Exception:  # noqa: BLE001 — 일부 버전은 dict 유사 객체
            logger.warning("dmd_time 연장 실패 — 예보가 학습 구간으로 제한됩니다.")
            n_forecast = self._n_train_steps
        self._n_forecast_steps = n_forecast

        self.training_metadata: dict[str, Any] = {}
        self._cache_mu: tuple[float, ...] | None = None
        self._cache_series: NDArray[np.float64] | None = None

    @property
    def t_max_forecast(self) -> float:
        """예보 가능한 물리 시간 상한."""
        return self._t0 + self._dt * (self._n_forecast_steps - 1)

    def _series_for(self, mu: tuple[float, ...]) -> NDArray[np.float64]:
        """μ 하나의 전체 시계열 (n_space, n_forecast_steps) — μ 단위로 캐시.

        ③Twin 의 t 슬라이더는 μ 고정 후 t 만 훑는 패턴이라, μ 캐시가 곧
        인터랙션 속도다 (reconstructed_data 는 μ 가 바뀔 때만 비싸다).
        """
        if self._cache_mu == mu and self._cache_series is not None:
            return self._cache_series
        self._pdmd.parameters = np.asarray([list(mu)], dtype=np.float64)
        rec = np.asarray(self._pdmd.reconstructed_data)
        series = np.real(rec[0])  # (n_space, n_time)
        self._cache_mu = mu
        self._cache_series = series
        return series

    def predict(self, params: Sequence[float]) -> NDArray[np.float64]:
        """``params = [μ₁..μₖ, t]`` 에서 필드를 예측한다 (t 는 마지막 원소).

        t 가 DMD 시간 격자 사이에 떨어지면 두 스텝을 선형 보간한다.
        """
        arr = np.asarray(params, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            raise ValueError(
                f"ParametricDMD 예측에는 (μ…, t) 최소 2개 값이 필요합니다. 현재: {arr.size}"
            )
        mu = tuple(float(v) for v in arr[:-1])
        t = float(arr[-1])
        series = self._series_for(mu)

        idx = (t - self._t0) / self._dt
        idx = min(max(idx, 0.0), float(series.shape[1] - 1))
        lo = int(np.floor(idx))
        hi = min(lo + 1, series.shape[1] - 1)
        w = idx - lo
        return (1.0 - w) * series[:, lo] + w * series[:, hi]
