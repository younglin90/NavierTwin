"""신경 연산자 추상 기반 클래스.

신경 연산자는 함수 공간 간의 매핑을 학습한다. 입출력이 딕셔너리 형태로
제공되어 다양한 연산자 구조(FNO, DeepONet 등)를 통일된 인터페이스로 지원한다.

Examples:
    커스텀 신경 연산자 구현::

        import numpy as np
        from naviertwin.core.operator_learning.base import BaseOperator

        class MyOperator(BaseOperator):
            def fit(self, dataset: dict) -> None:
                # dataset에는 "inputs", "outputs" 키가 포함됨
                inputs = dataset["inputs"]   # (n_samples, n_points, n_in_channels)
                outputs = dataset["outputs"] # (n_samples, n_points, n_out_channels)
                # 학습 로직 ...
                self.is_fitted = True

            def predict(self, inputs: dict) -> np.ndarray:
                self._check_fitted()
                x = inputs["x"]
                # 추론 로직 ...
                return result
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseOperator(ABC):
    """신경 연산자의 추상 기반 클래스.

    함수 공간 간의 연속 매핑을 학습한다. ``fit(dataset) → predict(inputs)``
    인터페이스를 제공하며, 입출력은 딕셔너리 형태를 사용한다.

    dataset 딕셔너리 규약 (fit 시):
        - ``"inputs"``: 입력 함수 스냅샷. shape = (n_samples, n_points, n_in_channels).
        - ``"outputs"``: 출력 함수 스냅샷. shape = (n_samples, n_points, n_out_channels).
        - ``"coords"``: 좌표. shape = (n_points, n_dims). 선택적.

    inputs 딕셔너리 규약 (predict 시):
        - ``"x"``: 쿼리 좌표 또는 입력 필드. 연산자 종류에 따라 다름.

    Attributes:
        is_fitted: :meth:`fit` 호출 후 True로 설정된다.
        n_epochs: 학습 에폭 수.
        device: PyTorch 학습 디바이스 문자열 ("cpu", "cuda", "cuda:0" 등).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.is_fitted: bool = False
        self.n_epochs: int = 0
        self.device: str = device

    @abstractmethod
    def fit(self, dataset: dict[str, Any]) -> None:
        """데이터셋으로 신경 연산자를 학습한다.

        Args:
            dataset: 학습 데이터 딕셔너리.
                필수 키: "inputs", "outputs".
                선택 키: "coords", "params", "masks".

        Raises:
            KeyError: 필수 키가 없는 경우.
            ValueError: 데이터 shape이 올바르지 않은 경우.
        """
        ...

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        """새로운 입력에 대한 출력 함수를 예측한다.

        Args:
            inputs: 예측 입력 딕셔너리.

        Returns:
            예측된 출력 필드. shape = (n_samples, n_points, n_out_channels)
            또는 (n_points, n_out_channels) (단일 샘플).

        Raises:
            RuntimeError: :meth:`fit`이 호출되지 않은 경우.
        """
        ...

    def save(self, path: str) -> None:
        """학습된 모델을 파일로 저장한다.

        Args:
            path: 저장할 파일 경로 (.pt 또는 .ckpt).

        Raises:
            RuntimeError: :meth:`fit`이 호출되지 않은 경우.
            NotImplementedError: 구체 클래스에서 구현되지 않은 경우.
        """
        self._check_fitted()
        raise NotImplementedError(f"{self.__class__.__name__}.save()가 구현되지 않았습니다.")

    def load(self, path: str) -> None:
        """파일에서 학습된 모델을 로드한다.

        Args:
            path: 로드할 파일 경로.

        Raises:
            FileNotFoundError: 파일이 없는 경우.
            NotImplementedError: 구체 클래스에서 구현되지 않은 경우.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.load()가 구현되지 않았습니다.")

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
            f"(device={self.device!r}, status={status})"
        )
