"""상태 공간 모델 추상 기반 클래스.

Mamba, S4 등 선형 순환 구조의 상태 공간 모델을 위한 공통 인터페이스.
``fit(dataset) → predict(inputs)`` 인터페이스를 제공한다.

Examples:
    커스텀 SSM 구현::

        import numpy as np
        from naviertwin.core.state_space.base import BaseSSM

        class MySSM(BaseSSM):
            def fit(self, dataset: dict) -> None:
                sequences = dataset["sequences"]  # (n_seqs, T, n_features)
                # 학습 ...
                self.is_fitted = True

            def predict(self, inputs: dict) -> np.ndarray:
                self._check_fitted()
                x0 = inputs["initial_state"]
                # 상태 전파 ...
                return predictions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseSSM(ABC):
    """상태 공간 모델의 추상 기반 클래스.

    선형/비선형 SSM (Mamba, S4, DeepOMamba 등)을 위한 공통 인터페이스.
    시계열 데이터를 입력으로 받아 미래 상태를 예측한다.

    Attributes:
        is_fitted: :meth:`fit` 호출 후 True로 설정된다.
        device: PyTorch 디바이스 문자열.
        state_dim: 상태 공간 차원.
        seq_len: 학습에 사용한 시퀀스 길이.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.is_fitted: bool = False
        self.device: str = device
        self.state_dim: int = 0
        self.seq_len: int = 0

    @abstractmethod
    def fit(self, dataset: dict[str, Any]) -> None:
        """시계열 데이터셋으로 SSM을 학습한다.

        Args:
            dataset: 학습 데이터 딕셔너리.
                필수 키: "sequences" — shape = (n_sequences, T, n_features).
                선택 키: "labels", "val_sequences", "epochs", "lr".

        Raises:
            KeyError: 필수 키가 없는 경우.
            ValueError: 데이터 shape이 올바르지 않은 경우.
        """
        ...

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        """초기 상태에서 시퀀스를 예측한다.

        Args:
            inputs: 예측 입력 딕셔너리.
                필수 키: "initial_state" — shape = (n_features,) 또는 (batch, n_features).
                선택 키: "n_steps" — 예측할 스텝 수 (기본값: seq_len).

        Returns:
            예측된 시퀀스. shape = (n_steps, n_features) 또는 (batch, n_steps, n_features).

        Raises:
            RuntimeError: :meth:`fit`이 호출되지 않은 경우.
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
            f"(device={self.device!r}, state_dim={self.state_dim}, status={status})"
        )
