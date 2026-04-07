"""대리 모델 추상 기반 클래스.

모든 대리 모델(RBF, Kriging, 신경망 기반 등)은 :class:`BaseSurrogate`를
상속하고 :meth:`fit` / :meth:`predict` 메서드를 구현해야 한다.

Examples:
    커스텀 대리 모델 구현::

        import numpy as np
        from naviertwin.core.surrogate.base import BaseSurrogate

        class LinearSurrogate(BaseSurrogate):
            def fit(self, X: np.ndarray, y: np.ndarray) -> None:
                self._coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                self.is_fitted = True

            def predict(self, X: np.ndarray) -> np.ndarray:
                self._check_fitted()
                return X @ self._coeffs
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseSurrogate(ABC):
    """대리 모델의 추상 기반 클래스.

    ``fit(X, y) → predict(X)`` 인터페이스를 강제한다.
    scikit-learn 스타일과 호환되도록 설계되었다.

    Attributes:
        is_fitted: :meth:`fit` 호출 후 True로 설정된다.
        input_dim: 입력 차원 수. fit 후 설정된다.
        output_dim: 출력 차원 수. fit 후 설정된다.
    """

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self.input_dim: int = 0
        self.output_dim: int = 0

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """학습 데이터로 대리 모델을 학습한다.

        Args:
            X: 입력 설계 변수 행렬. shape = (n_samples, n_features).
            y: 출력 응답값 행렬. shape = (n_samples,) 또는 (n_samples, n_outputs).

        Raises:
            ValueError: 입력 shape이 올바르지 않은 경우.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """새로운 입력에 대해 출력을 예측한다.

        Args:
            X: 예측할 입력 행렬. shape = (n_samples, n_features).

        Returns:
            예측된 출력값. shape = (n_samples,) 또는 (n_samples, n_outputs).

        Raises:
            RuntimeError: :meth:`fit`이 호출되지 않은 경우.
            ValueError: 입력 차원이 학습 시와 다른 경우.
        """
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """결정 계수(R²)로 모델 성능을 평가한다.

        Args:
            X: 테스트 입력 행렬. shape = (n_samples, n_features).
            y: 실제 출력값. shape = (n_samples,) 또는 (n_samples, n_outputs).

        Returns:
            R² 점수. 1.0이 최고, 0.0은 평균 예측과 동일, 음수는 나쁜 모델.

        Raises:
            RuntimeError: :meth:`fit`이 호출되지 않은 경우.
        """
        self._check_fitted()
        y_pred = self.predict(X)
        y_mean = np.mean(y, axis=0)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - ss_res / ss_tot)

    def get_params(self) -> dict[str, Any]:
        """모델 하이퍼파라미터를 딕셔너리로 반환한다.

        기본 구현은 빈 딕셔너리를 반환한다. 하위 클래스에서 재정의한다.

        Returns:
            하이퍼파라미터 딕셔너리.
        """
        return {}

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
            f"(input_dim={self.input_dim}, output_dim={self.output_dim}, status={status})"
        )
