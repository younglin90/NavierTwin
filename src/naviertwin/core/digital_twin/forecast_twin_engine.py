"""POD + 시계열 예보기 트윈 엔진 — DMD 옆의 대안 예보 백엔드 공용 래퍼.

``core/time_series/`` 와 ``core/operator_learning/koopman/`` 에는 이미
``BaseTimeSeries`` 계약(``fit({"sequences": (N,T,F)}) → predict(initial_state,
n_steps) -> (n_steps,F)``)으로 완성된 예보기 4종(LSTM/KNO/LatentDynamics/
NeuralODE)이 있었지만 트윈 예측 파이프라인에는 하나도 배선돼 있지 않았다.
DMD(:class:`naviertwin.core.digital_twin.dmd_engine.DMDTwinEngine`)와 같은
"학습 구간 밖 t 예보" 계약을 이 4종에도 주기 위한 어댑터가 이 모듈이다.

DMD 는 스냅샷을 직접 다루지만(저랭크 선형 동역학 가정), 여기 예보기들은
비선형 표현력이 있는 대신 상태 차원이 커지면(공간 점 수) 학습이 무겁다 —
그래서 기존 ROM 이 쓰는 :class:`SnapshotPOD` 로 필드를 POD 계수로 먼저
압축하고, 예보기는 그 **계수 시계열**의 시간 전이만 학습한다. 예측 시엔
계수를 롤아웃한 뒤 POD 역변환(``decode``)으로 필드를 복원한다.

두 가지 학습 모드:

단일 케이스(시계열, 문제 유형 A)
    :meth:`for_single_case` 로 구성한다. 학습 구간 안(t ≤ t_train_max)은
    POD 인코딩한 **실측** 계수를 그대로 쓰고, 그 밖은 예보기가 마지막
    lookback 윈도우에서 자기회귀 롤아웃한 값을 이어 붙인다 — DMD 의
    ``reconstruct(t)`` 와 같은 "학습 구간=실측, 밖=예보" 의미론이다.

케이스 세트(비정상 파라미터 스윕, 문제 유형 B)
    :meth:`for_case_set` 로 구성한다. **모든 케이스의 계수 시퀀스를 하나의
    예보기에 합쳐 학습**한다(공유 동역학 가정) — DMD 의 partitioned(케이스별
    독립 모델) 방식과 다른 선택이다. μ 조건부는 "케이스 초기 lookback
    윈도우"를 :class:`RBFSurrogate` 로 μ 보간해 반영한다 — 즉 **새 μ 는
    다른 초기조건으로 치환**된다. 케이스별 완전 독립 모델이 아니므로 μ 축의
    국소적 동역학 차이(예: 케이스마다 지배 주파수가 다름)를 신경망이 공유
    가중치로 충분히 표현할 만큼 학습 데이터가 있을 때만 잘 맞는다 — "표현력
    vs 데이터효율" 트레이드오프를 진 것이며, 이는 ``reconstruction_error``
    로 학습 후 확인해야 한다(DMD 와 같은 트래픽 라이트 UI 를 공유한다).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["TimeSeriesForecastTwinEngine"]


def _n_forecast_steps(n_known: int, forecast_factor: float) -> int:
    """학습 구간 스텝 수에서 예보 상한 스텝 수를 계산한다 (ParametricDMD와 같은 공식)."""
    return max(n_known, int(round((n_known - 1) * forecast_factor)) + 1)


class TimeSeriesForecastTwinEngine:
    """LSTM/KNO/LatentODE/NeuralODE 예보기를 트윈 엔진 계약으로 감싼다.

    Attributes:
        reducer: 학습 완료된 ``SnapshotPOD`` (필드 ↔ 계수 변환).
        forecaster: 학습 완료된 ``BaseTimeSeries`` 구현체.
        backend: 백엔드 식별자 (``lstm``/``koopman_no``/``latent_ode``/
            ``neural_ode``).
        case_mode: ``True`` 면 케이스 세트(μ 조건부), ``False`` 면 단일
            케이스(시계열) 모드.
        training_metadata: 학습 범위·필드 등 (웹 GUI 가 슬라이더 구성에 사용).
    """

    def __init__(
        self,
        reducer: Any,
        forecaster: Any,
        *,
        backend: str,
        t0: float,
        dt: float,
        forecast_factor: float = 1.5,
        case_mode: bool = False,
    ) -> None:
        if not hasattr(forecaster, "predict") or not hasattr(forecaster, "fit"):
            raise TypeError("forecaster must expose fit()/predict() (BaseTimeSeries)")
        self.reducer = reducer
        self.forecaster = forecaster
        self.model = forecaster  # 공통 접근자 (다른 엔진들과의 덕타이핑 일관성)
        self.backend = str(backend)
        self.lookback = max(1, int(getattr(forecaster, "lookback", 1)))
        self.t0 = float(t0)
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt 는 양수여야 합니다.")
        self.forecast_factor = float(forecast_factor)
        self.case_mode = bool(case_mode)

        self.reducer_type = "pod_forecast"
        self.surrogate_type = self.backend
        self.model_type = f"forecast_{self.backend}"
        self.training_metadata: dict[str, Any] = {}

        # 단일 케이스 모드 상태 — for_single_case 가 채운다.
        self._full_series: NDArray[np.float64] | None = None
        self._n_forecast_steps: int = 0

        # 케이스 세트 모드 상태 — for_case_set 이 채운다.
        self._mu_interp: Any | None = None
        self._cache_mu: tuple[float, ...] | None = None
        self._cache_series: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # 생성자
    # ------------------------------------------------------------------

    @classmethod
    def for_single_case(
        cls,
        reducer: Any,
        forecaster: Any,
        *,
        backend: str,
        train_coeffs: NDArray[np.float64],
        t0: float,
        dt: float,
        forecast_factor: float = 1.5,
    ) -> "TimeSeriesForecastTwinEngine":
        """단일 케이스(시계열) 트윈 — 학습 구간은 실측, 밖은 예보기 롤아웃.

        Args:
            train_coeffs: POD 인코딩된 학습 계수 시퀀스. shape=(n_train, n_modes).
        """
        engine = cls(
            reducer, forecaster, backend=backend, t0=t0, dt=dt,
            forecast_factor=forecast_factor, case_mode=False,
        )
        known = np.asarray(train_coeffs, dtype=np.float64)
        n_known = known.shape[0]
        n_forecast = _n_forecast_steps(n_known, forecast_factor)
        n_extra = n_forecast - n_known
        if n_extra <= 0:
            series = known[:n_forecast].copy()
        else:
            window = known[-engine.lookback :]
            extra = np.asarray(forecaster.predict(window, n_extra), dtype=np.float64)
            series = np.concatenate([known, extra], axis=0)
        engine._full_series = series
        engine._n_forecast_steps = series.shape[0]
        return engine

    @classmethod
    def for_case_set(
        cls,
        reducer: Any,
        forecaster: Any,
        *,
        backend: str,
        mu_interp: Any,
        n_train_steps: int,
        t0: float,
        dt: float,
        forecast_factor: float = 1.5,
    ) -> "TimeSeriesForecastTwinEngine":
        """케이스 세트(비정상 스윕) 트윈 — μ 는 초기조건 보간으로 조건부화.

        Args:
            mu_interp: 학습 완료된 ``RBFSurrogate`` — μ → 평탄화된 초기
                lookback 윈도우(``(lookback*n_modes,)``)를 예측한다.
            n_train_steps: 케이스당 타임스텝 수 (모든 케이스 동일 가정).
        """
        engine = cls(
            reducer, forecaster, backend=backend, t0=t0, dt=dt,
            forecast_factor=forecast_factor, case_mode=True,
        )
        engine._mu_interp = mu_interp
        engine._n_forecast_steps = _n_forecast_steps(int(n_train_steps), forecast_factor)
        return engine

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------

    @property
    def t_max_forecast(self) -> float:
        """예보 가능한 물리 시간 상한."""
        return self.t0 + self.dt * (self._n_forecast_steps - 1)

    @property
    def input_dim(self) -> int:
        """(μ…, t) 입력 차원 — 데스크톱 Twin 패널이 스핀박스 수를 맞추는 데 쓴다."""
        names = self.training_metadata.get("param_names")
        if isinstance(names, list) and names:
            return len(names)
        return 2 if self.case_mode else 1

    def _series_for_mu(self, mu: tuple[float, ...]) -> NDArray[np.float64]:
        """μ 하나의 전체 계수 궤적 — μ 단위로 캐시(반복 t 슬라이더 스캔이 흔한 패턴)."""
        if self._cache_mu == mu and self._cache_series is not None:
            return self._cache_series
        if self._mu_interp is None:
            raise RuntimeError("케이스 세트 트윈인데 mu_interp 가 없습니다.")
        flat = np.asarray(
            self._mu_interp.predict(np.asarray(mu, dtype=np.float64).reshape(1, -1)),
            dtype=np.float64,
        ).reshape(-1)
        n_modes = flat.size // self.lookback
        window = flat.reshape(self.lookback, n_modes)
        n_extra = self._n_forecast_steps - self.lookback
        if n_extra <= 0:
            series = window[: self._n_forecast_steps]
        else:
            extra = np.asarray(
                self.forecaster.predict(window, n_extra), dtype=np.float64
            )
            series = np.concatenate([window, extra], axis=0)
        self._cache_mu = mu
        self._cache_series = series
        return series

    def predict(self, params: Any) -> NDArray[np.float64]:
        """단일 케이스면 ``[t]``, 케이스 세트면 ``[μ₁..μₖ, t]`` 에서 필드를 예측한다.

        t 가 시간 격자 사이에 떨어지면 이웃 두 스텝을 선형 보간한다(DMD 계열과
        같은 방식).
        """
        arr = np.asarray(params, dtype=np.float64).reshape(-1)
        if self.case_mode:
            if arr.size < 2:
                raise ValueError(
                    f"케이스 세트 예보에는 (μ…, t) 최소 2개 값이 필요합니다. 현재: {arr.size}"
                )
            mu = tuple(float(v) for v in arr[:-1])
            t = float(arr[-1])
            series = self._series_for_mu(mu)
        else:
            if self._full_series is None:
                raise RuntimeError("엔진이 학습되지 않았습니다.")
            t = float(arr[-1])
            series = self._full_series

        idx = (t - self.t0) / self.dt
        idx = min(max(idx, 0.0), float(series.shape[0] - 1))
        lo = int(np.floor(idx))
        hi = min(lo + 1, series.shape[0] - 1)
        w = idx - lo
        coeff = (1.0 - w) * series[lo] + w * series[hi]

        field = self.reducer.decode(coeff[np.newaxis, :])
        return np.asarray(field, dtype=np.float64)[:, 0]

    # ------------------------------------------------------------------
    # 저장 / 로드
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """엔진을 pickle 로 저장한다."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "TimeSeriesForecastTwinEngine":
        """저장된 예보 트윈 엔진을 로드한다."""
        with Path(path).open("rb") as f:
            engine = pickle.load(f)
        if not isinstance(engine, cls):
            raise TypeError(f"TimeSeriesForecastTwinEngine 파일이 아닙니다: {path}")
        return engine

    def get_params(self) -> dict[str, Any]:
        """TwinEngine 호출자와 호환되는 메타데이터."""
        return {
            "reducer_type": self.reducer_type,
            "surrogate_type": self.surrogate_type,
            "model_type": self.model_type,
            "backend": self.backend,
            "case_mode": self.case_mode,
        }
