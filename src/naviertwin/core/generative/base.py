"""생성 모델 추상 기반 클래스.

확산 모델, 조건부 생성 모델 등 생성 모델의 공통 인터페이스를 정의한다.
``fit(dataset) → generate(n_samples, condition)`` 인터페이스를 강제한다.

Examples:
    커스텀 생성 모델 구현::

        import numpy as np
        from naviertwin.core.generative.base import BaseGenerative

        class MyDiffusionModel(BaseGenerative):
            def fit(self, dataset: dict) -> None:
                data = dataset["snapshots"]  # (n_samples, n_features)
                # 확산 학습 ...
                self.is_fitted = True

            def generate(self, n_samples: int, condition: dict | None = None) -> np.ndarray:
                self._check_fitted()
                # 생성 로직 (역방향 확산 과정) ...
                return generated  # (n_samples, n_features)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseGenerative(ABC):
    """생성 모델의 추상 기반 클래스.

    유동장 스냅샷, 메쉬 데이터 등을 생성하는 모델을 위한 공통 인터페이스.
    ``fit → generate`` 인터페이스를 강제한다.

    Attributes:
        is_fitted: :meth:`fit` 호출 후 True로 설정된다.
        device: PyTorch 디바이스 문자열.
        latent_dim: 잠재 공간 차원.
        data_shape: 생성되는 샘플의 shape (n_features 또는 (H, W, C) 등).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.is_fitted: bool = False
        self.device: str = device
        self.latent_dim: int = 0
        self.data_shape: tuple[int, ...] = (0,)

    @abstractmethod
    def fit(self, dataset: dict[str, Any]) -> None:
        """데이터셋으로 생성 모델을 학습한다.

        Args:
            dataset: 학습 데이터 딕셔너리.
                필수 키: "snapshots" — shape = (n_samples, *data_shape).
                선택 키: "labels", "val_snapshots", "epochs", "lr".

        Raises:
            KeyError: 필수 키가 없는 경우.
            ValueError: 데이터 shape이 올바르지 않은 경우.
        """
        ...

    @abstractmethod
    def generate(
        self,
        n_samples: int,
        condition: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """새로운 샘플을 생성한다.

        Args:
            n_samples: 생성할 샘플 수.
            condition: 조건 딕셔너리 (조건부 생성 시). None이면 무조건 생성.
                예: {"Re": 1000.0, "AoA": 5.0}.

        Returns:
            생성된 샘플. shape = (n_samples, *data_shape).

        Raises:
            RuntimeError: :meth:`fit`이 호출되지 않은 경우.
            ValueError: n_samples가 0 이하인 경우.
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
            f"(device={self.device!r}, latent_dim={self.latent_dim}, status={status})"
        )
