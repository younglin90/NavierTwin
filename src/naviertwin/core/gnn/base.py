"""그래프 신경망 추상 기반 클래스.

모든 GNN 기반 모델은 :class:`BaseGNN`을 상속하고
:meth:`fit` / :meth:`predict` 메서드를 구현해야 한다.

그래프 데이터는 PyTorch Geometric 의 ``Data`` 또는 ``HeteroData`` 객체를
사용하지만, 의존성 없이 실행될 수 있도록 ``Any`` 타입으로 선언한다.

Examples:
    커스텀 GNN 구현::

        import numpy as np
        from naviertwin.core.gnn.base import BaseGNN

        class MyGNN(BaseGNN):
            def fit(self, dataset: dict) -> None:
                # PyTorch Geometric 데이터 로더로 학습
                self.is_fitted = True

            def predict(self, inputs: dict) -> np.ndarray:
                self._check_fitted()
                # 그래프 추론 ...
                return result
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseGNN(ABC):
    """그래프 신경망의 추상 기반 클래스.

    메쉬를 그래프로 변환하여 학습하는 모델을 위한 공통 인터페이스를 제공한다.
    ``fit(dataset) → predict(inputs)`` 인터페이스를 강제한다.

    Attributes:
        is_fitted: :meth:`fit` 호출 후 True로 설정된다.
        device: PyTorch 디바이스 문자열.
        n_node_features: 노드 특성 차원.
        n_edge_features: 엣지 특성 차원.
        n_output_features: 출력 특성 차원.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.is_fitted: bool = False
        self.device: str = device
        self.n_node_features: int = 0
        self.n_edge_features: int = 0
        self.n_output_features: int = 0

    @abstractmethod
    def fit(self, dataset: dict[str, Any]) -> None:
        """그래프 데이터셋으로 GNN을 학습한다.

        Args:
            dataset: 학습 데이터 딕셔너리.
                필수 키: "graphs" (PyG Data 객체 목록 또는 DataLoader).
                선택 키: "val_graphs", "epochs", "lr".

        Raises:
            KeyError: 필수 키가 없는 경우.
            ImportError: torch-geometric이 설치되지 않은 경우.
        """
        ...

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        """그래프 입력에 대한 노드/엣지/그래프 수준 출력을 예측한다.

        Args:
            inputs: 예측 입력 딕셔너리.
                필수 키: "graph" (PyG Data 객체 또는 그에 상응하는 딕셔너리).

        Returns:
            예측된 출력. shape = (n_nodes, n_output_features) (노드 수준)
            또는 (n_graphs, n_output_features) (그래프 수준).

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
            f"(device={self.device!r}, status={status})"
        )
