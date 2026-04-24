"""Streaming Digital Twin — 실시간 관측 → EnKF 동화 → 예측.

센서 관측이 스트림으로 도착할 때, EnKF 로 앙상블을 업데이트하고
현재 상태 추정치를 유지한다.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.digital_twin.streaming_twin import StreamingDigitalTwin
    >>> rng = np.random.default_rng(0)
    >>> # 선형 시스템 x_{t+1} = A x_t + η
    >>> A = np.array([[0.9, 0.05], [0.02, 0.95]])
    >>> H = np.eye(2)
    >>> R = 0.02 * np.eye(2)
    >>> twin = StreamingDigitalTwin(
    ...     state_dim=2, n_ensemble=40,
    ...     model_fn=lambda x: A @ x,
    ...     H=H, R=R, rng=rng,
    ... )
    >>> twin.initialize(rng.standard_normal((40, 2)))
    >>> # 시뮬레이션된 관측
    >>> for _ in range(5):
    ...     twin.step()
    ...     twin.assimilate(np.array([0.5, -0.3]) + 0.01 * rng.standard_normal(2))
    >>> est = twin.estimate()
    >>> est.shape
    (2,)
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.data_assimilation.enkf import EnKF
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class StreamingDigitalTwin:
    """Model-forecast + EnKF 동화 기반 실시간 상태 추정."""

    def __init__(
        self,
        state_dim: int,
        n_ensemble: int,
        model_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        H: NDArray[np.float64],
        R: NDArray[np.float64],
        process_noise: float = 0.01,
        history_size: int = 100,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.n_ensemble = n_ensemble
        self.model_fn = model_fn
        self.process_noise = process_noise
        self.rng = rng or np.random.default_rng()
        self.enkf = EnKF(H=H, R=R)
        self.history: deque = deque(maxlen=history_size)
        self._ensemble: NDArray[np.float64] | None = None

    def initialize(self, ensemble: NDArray[np.float64]) -> None:
        ens = np.asarray(ensemble, dtype=np.float64)
        if ens.shape != (self.n_ensemble, self.state_dim):
            raise ValueError(
                f"ensemble shape={ens.shape}, 기대=({self.n_ensemble},{self.state_dim})"
            )
        self._ensemble = ens.copy()
        self.history.append(self.estimate())
        logger.info("StreamingDigitalTwin 초기화: N=%d", self.n_ensemble)

    def step(self) -> None:
        """앙상블을 1 스텝 전파 (모델 + 프로세스 잡음)."""
        if self._ensemble is None:
            raise RuntimeError("initialize() 먼저 호출")
        new_ens = np.zeros_like(self._ensemble)
        for i in range(self.n_ensemble):
            new_ens[i] = self.model_fn(self._ensemble[i])
        noise = self.rng.standard_normal(new_ens.shape) * self.process_noise
        self._ensemble = new_ens + noise
        self.history.append(self.estimate())

    def assimilate(self, observation: NDArray[np.float64]) -> None:
        if self._ensemble is None:
            raise RuntimeError("initialize() 먼저 호출")
        self._ensemble = self.enkf.analysis(
            self._ensemble, observation, rng=self.rng
        )
        self.history.append(self.estimate())

    def estimate(self) -> NDArray[np.float64]:
        if self._ensemble is None:
            raise RuntimeError("initialize() 먼저 호출")
        return self._ensemble.mean(axis=0)

    def uncertainty(self) -> NDArray[np.float64]:
        if self._ensemble is None:
            raise RuntimeError("initialize() 먼저 호출")
        return self._ensemble.std(axis=0)


__all__ = ["StreamingDigitalTwin"]
