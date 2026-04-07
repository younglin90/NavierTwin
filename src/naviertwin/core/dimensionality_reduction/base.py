"""차원 축소기 추상 기반 클래스.

:class:`BaseReducer`를 상속하여 POD, SVD, 오토인코더 등
다양한 차원 축소 기법을 일관된 인터페이스로 구현한다.

Examples:
    커스텀 축소기 구현::

        import numpy as np
        from naviertwin.core.dimensionality_reduction.base import BaseReducer

        class MyReducer(BaseReducer):
            def fit(self, snapshots: np.ndarray) -> None:
                # 학습 로직
                ...

            def encode(self, snapshots: np.ndarray) -> np.ndarray:
                # 인코딩 로직
                ...

            def decode(self, coefficients: np.ndarray) -> np.ndarray:
                # 디코딩 로직
                ...

            @property
            def energy_ratio(self) -> np.ndarray:
                # 에너지 비율 반환
                ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseReducer(ABC):
    """차원 축소기의 추상 기반 클래스.

    ``fit → encode → decode`` 인터페이스를 강제한다.
    선형(POD, 랜덤화 SVD 등)과 비선형(AE, VAE 등) 모두 이 인터페이스를 구현한다.

    Attributes:
        is_fitted: fit()이 호출된 후 True로 설정된다.
        n_components: 사용하는 기저(모드/잠재 차원) 수. fit 후 설정.
    """

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self.n_components: int = 0

    @abstractmethod
    def fit(self, snapshots: np.ndarray) -> None:
        """스냅샷 데이터로 차원 축소 모델을 학습한다.

        Args:
            snapshots: 스냅샷 행렬. shape = (n_samples, n_features) 또는
                (n_features, n_samples) — 구체 클래스에서 규약을 명시.

        Raises:
            ValueError: 입력 차원이 올바르지 않은 경우.
        """
        ...

    @abstractmethod
    def encode(self, snapshots: np.ndarray) -> np.ndarray:
        """스냅샷을 저차원 계수(잠재 공간)로 인코딩한다.

        Args:
            snapshots: 인코딩할 스냅샷. shape = (n_samples, n_features).

        Returns:
            저차원 계수 행렬. shape = (n_samples, n_components).

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        ...

    @abstractmethod
    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        """저차원 계수를 원래 차원의 스냅샷으로 디코딩(재구성)한다.

        Args:
            coefficients: 잠재 계수 행렬. shape = (n_samples, n_components).

        Returns:
            재구성된 스냅샷. shape = (n_samples, n_features).

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        ...

    def reconstruct(self, snapshots: np.ndarray, n_modes: int | None = None) -> np.ndarray:
        """스냅샷을 인코딩 후 디코딩하여 재구성한다.

        encode → decode를 순서대로 호출하는 편의 메서드.
        ``n_modes``를 지정하면 상위 n개 모드만 사용해 재구성한다.

        Args:
            snapshots: 재구성할 스냅샷. shape = (n_samples, n_features).
            n_modes: 사용할 모드(기저) 수. None이면 전체 모드 사용.

        Returns:
            재구성된 스냅샷. shape = (n_samples, n_features).

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
            ValueError: n_modes가 n_components보다 큰 경우.
        """
        if not self.is_fitted:
            raise RuntimeError("reconstruct() 전에 fit()을 먼저 호출해야 합니다.")
        if n_modes is not None and n_modes > self.n_components:
            raise ValueError(
                f"n_modes({n_modes})는 n_components({self.n_components})보다 클 수 없습니다."
            )

        coefficients = self.encode(snapshots)
        if n_modes is not None:
            # 상위 n_modes 이후의 계수를 0으로 마스킹
            coefficients = coefficients.copy()
            coefficients[:, n_modes:] = 0.0
        return self.decode(coefficients)

    @property
    @abstractmethod
    def energy_ratio(self) -> np.ndarray:
        """각 모드(기저)의 누적 에너지 비율을 반환한다.

        Returns:
            누적 에너지 비율 배열. shape = (n_components,).
            마지막 원소는 1.0에 가까워야 한다.

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
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
        return f"{self.__class__.__name__}(n_components={self.n_components}, status={status})"
