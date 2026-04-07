"""RBF (Radial Basis Function) Surrogate.

SMT 라이브러리의 RBF 서로게이트를 래핑한다.
SMT 미설치 시 sklearn LinearRegression으로 자동 폴백한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.surrogate.base import BaseSurrogate
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class RBFSurrogate(BaseSurrogate):
    """SMT 라이브러리 RBF 서로게이트.

    소량 샘플에서도 동작하며 보간 특성을 가진다.
    SMT 미설치 시 sklearn LinearRegression으로 자동 폴백한다.

    Attributes:
        d0: RBF 폭 파라미터.

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate
        >>> rng = np.random.default_rng(0)
        >>> X = rng.uniform(-1, 1, (30, 2))
        >>> y = np.sin(X[:, 0]) + X[:, 1] ** 2
        >>> rbf = RBFSurrogate(d0=1.0)
        >>> rbf.fit(X, y)
        >>> y_pred = rbf.predict(X)
    """

    def __init__(self, d0: float = 1.0) -> None:
        """초기화.

        Args:
            d0: RBF 폭 파라미터 (기본: 1.0).
                값이 클수록 넓은 영향 범위를 가진다.
        """
        super().__init__()
        self.d0 = d0
        self._model: Any = None
        self._backend: str = "unknown"

    # ------------------------------------------------------------------
    # BaseSurrogate 추상 메서드 구현
    # ------------------------------------------------------------------

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """학습 데이터로 RBF 서로게이트를 학습한다.

        Args:
            X: 입력 설계 변수. shape = (n_samples, n_features).
            y: 출력 응답값. shape = (n_samples,) 또는 (n_samples, n_outputs).

        Raises:
            ValueError: 입력 shape이 올바르지 않은 경우.
        """
        if X.ndim != 2:
            raise ValueError(f"X는 2D 배열이어야 합니다. 현재 shape: {X.shape}")

        y_2d = y if y.ndim == 2 else y[:, np.newaxis]
        n_samples, n_features = X.shape
        n_outputs = y_2d.shape[1]

        self.input_dim = n_features
        self.output_dim = n_outputs

        logger.debug(
            "RBFSurrogate.fit: n_samples=%d, n_features=%d, n_outputs=%d",
            n_samples,
            n_features,
            n_outputs,
        )

        if self._try_fit_smt(X, y_2d):
            self._backend = "smt"
        else:
            # _fit_sklearn_fallback 내부에서 _backend를 설정함
            self._fit_sklearn_fallback(X, y_2d)

        self.is_fitted = True
        logger.info("RBFSurrogate 학습 완료 (backend=%s).", self._backend)

    def _try_fit_smt(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> bool:
        """SMT RBF로 학습을 시도한다.

        Args:
            X: 입력 데이터. shape = (n_samples, n_features).
            y: 출력 데이터. shape = (n_samples, n_outputs).

        Returns:
            성공 시 True, 실패 시 False.
        """
        try:
            from smt.surrogate_models import RBF  # type: ignore[import]

            models = []
            for i in range(y.shape[1]):
                sm = RBF(d0=self.d0, print_global=False)
                sm.set_training_values(X, y[:, i])
                sm.train()
                models.append(sm)
            self._model = models
            return True
        except ImportError:
            logger.warning("smt 미설치 — sklearn LinearRegression으로 폴백합니다.")
            return False
        except Exception as exc:
            logger.warning("SMT RBF 학습 실패: %s — sklearn으로 폴백합니다.", exc)
            return False

    def _fit_sklearn_fallback(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> None:
        """sklearn LinearRegression 또는 numpy 최소 제곱 폴백 학습.

        sklearn → numpy lstsq 순서로 시도한다.

        Args:
            X: 입력 데이터. shape = (n_samples, n_features).
            y: 출력 데이터. shape = (n_samples, n_outputs).
        """
        try:
            from sklearn.linear_model import LinearRegression  # type: ignore[import]

            model = LinearRegression()
            model.fit(X, y)
            self._model = model
            self._backend = "sklearn"
            return
        except ImportError:
            pass

        # numpy lstsq 최종 폴백
        logger.warning("sklearn 미설치 — numpy lstsq 최소 제곱 폴백을 사용합니다.")
        # 편향 항 추가
        X_aug = np.column_stack([X, np.ones(X.shape[0])])
        coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        self._model = {"coeffs": coeffs, "type": "numpy_lstsq"}
        self._backend = "numpy"

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """새로운 입력에 대해 출력을 예측한다.

        Args:
            X: 예측할 입력. shape = (n_samples, n_features).

        Returns:
            예측값. shape = (n_samples, n_outputs) 또는 (n_samples,).

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        self._check_fitted()

        if X.ndim == 1:
            X = X[np.newaxis, :]

        if self._backend == "smt":
            preds = np.column_stack(
                [sm.predict_values(X).ravel() for sm in self._model]
            )
        elif self._backend == "sklearn":
            preds = self._model.predict(X)
            if preds.ndim == 1:
                preds = preds[:, np.newaxis]
        else:
            # numpy lstsq 폴백
            X_aug = np.column_stack([X, np.ones(X.shape[0])])
            coeffs = self._model["coeffs"]
            preds = X_aug @ coeffs

        if preds.ndim == 1:
            preds = preds[:, np.newaxis]

        if preds.shape[1] == 1:
            return preds.ravel()
        return preds

    def score(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        """R² 점수로 모델 성능을 평가한다.

        Args:
            X: 테스트 입력. shape = (n_samples, n_features).
            y: 실제 출력값.

        Returns:
            R² 점수.
        """
        return super().score(X, y)

    def get_params(self) -> dict[str, Any]:
        """하이퍼파라미터를 반환한다.

        Returns:
            {"d0": ..., "backend": ...}
        """
        return {"d0": self.d0, "backend": self._backend}
