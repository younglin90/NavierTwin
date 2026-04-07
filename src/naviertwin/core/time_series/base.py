"""시계열 예측 모델 추상 기반 클래스.

LSTM, 트랜스포머, Neural ODE 등 시계열 기반 유동장 예측 모델의
공통 인터페이스를 정의한다.

Examples:
    커스텀 시계열 모델 구현::

        import numpy as np
        from naviertwin.core.time_series.base import BaseTimeSeries

        class MyLSTM(BaseTimeSeries):
            def fit(self, dataset: dict) -> None:
                seqs = dataset["sequences"]  # (n_seqs, T, n_features)
                # 학습 ...
                self.is_fitted = True

            def predict(self, initial_state: np.ndarray, n_steps: int) -> np.ndarray:
                self._check_fitted()
                # 롤아웃 예측 ...
                return predictions  # (n_steps, n_features)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseTimeSeries(ABC):
    """시계열 예측 모델의 추상 기반 클래스.

    유동장의 시간 발전을 예측하는 모델을 위한 공통 인터페이스.
    ``fit(dataset) → predict(initial_state, n_steps)`` 인터페이스를 강제한다.

    Attributes:
        is_fitted: :meth:`fit` 호출 후 True로 설정된다.
        device: PyTorch 디바이스 문자열.
        n_features: 상태 벡터 차원.
        lookback: 예측에 사용하는 과거 스텝 수 (lookback window).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.is_fitted: bool = False
        self.device: str = device
        self.n_features: int = 0
        self.lookback: int = 1

    @abstractmethod
    def fit(self, dataset: dict[str, Any]) -> None:
        """시계열 데이터셋으로 모델을 학습한다.

        Args:
            dataset: 학습 데이터 딕셔너리.
                필수 키: "sequences" — shape = (n_sequences, T, n_features).
                선택 키: "val_sequences", "epochs", "lr", "batch_size".

        Raises:
            KeyError: 필수 키가 없는 경우.
            ValueError: 데이터 shape이 올바르지 않은 경우.
        """
        ...

    @abstractmethod
    def predict(self, initial_state: np.ndarray, n_steps: int) -> np.ndarray:
        """초기 상태에서 n_steps 앞의 상태를 자기회귀(auto-regressive) 방식으로 예측한다.

        Args:
            initial_state: 초기 상태 벡터. shape = (n_features,) 또는
                lookback window를 포함하는 경우 (lookback, n_features).
            n_steps: 예측할 미래 타임스텝 수.

        Returns:
            예측된 시계열. shape = (n_steps, n_features).

        Raises:
            RuntimeError: :meth:`fit`이 호출되지 않은 경우.
            ValueError: n_steps가 1 미만인 경우.
        """
        ...

    def _check_fitted(self) -> None:
        """fit()이 완료되었는지 확인한다.

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}의 fit()을 먼저 호출해야 합니다."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}"
            f"(device={self.device!r}, n_features={self.n_features}, status={status})"
        )
