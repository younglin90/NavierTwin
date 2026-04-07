"""등변 신경망 추상 기반 클래스.

회전, 반사, 평행이동 등의 물리적 대칭성을 보존하는 신경망 모델의
공통 인터페이스를 정의한다. e3nn, escnn 기반 모델이 이를 상속한다.

Examples:
    커스텀 등변 모델 구현::

        import numpy as np
        from naviertwin.core.equivariant.base import BaseEquivariant

        class GroupEquivFNO(BaseEquivariant):
            def fit(self, dataset: dict) -> None:
                # SO(3) 등변 학습 ...
                self.is_fitted = True
                self.symmetry_group = "SO(3)"

            def predict(self, inputs: dict) -> np.ndarray:
                self._check_fitted()
                # 등변 추론 ...
                return result
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEquivariant(ABC):
    """등변 신경망의 추상 기반 클래스.

    물리적 대칭성(회전, 반사, 병진 불변성 등)을 보존하도록 설계된
    신경망 모델을 위한 공통 인터페이스.
    ``fit(dataset) → predict(inputs)`` 인터페이스를 강제한다.

    Attributes:
        is_fitted: :meth:`fit` 호출 후 True로 설정된다.
        device: PyTorch 디바이스 문자열.
        symmetry_group: 보존하는 대칭 그룹 이름. 예: "SO(3)", "E(3)", "O(3)".
    """

    def __init__(self, device: str = "cpu") -> None:
        self.is_fitted: bool = False
        self.device: str = device
        self.symmetry_group: str = ""

    @abstractmethod
    def fit(self, dataset: dict[str, Any]) -> None:
        """데이터셋으로 등변 모델을 학습한다.

        학습 시 대칭 그룹에 대한 등변성 제약이 자동으로 적용된다.

        Args:
            dataset: 학습 데이터 딕셔너리.
                필수 키: "inputs", "outputs".
                선택 키: "coords", "epochs", "lr".

        Raises:
            KeyError: 필수 키가 없는 경우.
            ImportError: e3nn 또는 escnn이 설치되지 않은 경우.
        """
        ...

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        """등변성을 보존하며 출력을 예측한다.

        입력 데이터에 대칭 변환이 적용되면 출력에도 동일한 변환이
        자동으로 적용된다 (등변성 보장).

        Args:
            inputs: 예측 입력 딕셔너리.

        Returns:
            예측된 출력값.

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
            f"(symmetry_group={self.symmetry_group!r}, device={self.device!r}, "
            f"status={status})"
        )
