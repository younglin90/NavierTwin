"""Kriging (Gaussian Process) Surrogate.

SMT 라이브러리의 Kriging 서로게이트를 래핑하며
불확실성 정량화(예측 분산)를 제공한다.
SMT 미설치 시 sklearn GaussianProcessRegressor로 폴백하고,
sklearn도 없으면 LinearRegression으로 최종 폴백한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.surrogate.base import BaseSurrogate
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class KrigingSurrogate(BaseSurrogate):
    """SMT 라이브러리 Kriging (KPLS 포함) 서로게이트.

    불확실성 정량화(예측 분산)를 제공한다.
    SMT 미설치 시 sklearn GaussianProcessRegressor로 자동 폴백한다.

    Attributes:
        corr: 상관 함수 타입 ('squar_exp', 'abs_exp', 'matern52').
        poly: 추세 함수 타입 ('constant', 'linear', 'quadratic').

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.surrogate.kriging_surrogate import KrigingSurrogate
        >>> rng = np.random.default_rng(0)
        >>> X = rng.uniform(-1, 1, (20, 2))
        >>> y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        >>> krig = KrigingSurrogate(corr="squar_exp", poly="constant")
        >>> krig.fit(X, y)
        >>> y_pred = krig.predict(X)
        >>> y_pred, y_var = krig.predict_with_variance(X)
    """

    def __init__(
        self,
        corr: str = "squar_exp",
        poly: str = "constant",
    ) -> None:
        """초기화.

        Args:
            corr: 상관 함수 타입.
                'squar_exp' (제곱 지수), 'abs_exp' (절대 지수),
                'matern52' (Matern 5/2).
            poly: 추세 함수 타입.
                'constant', 'linear', 'quadratic'.
        """
        super().__init__()
        self.corr = corr
        self.poly = poly
        self._model: Any = None
        self._backend: str = "unknown"
        self._numpy_coeffs: Any = None   # numpy lstsq 폴백 계수
        self._linear_model: Any = None   # sklearn LinearRegression 폴백

    # ------------------------------------------------------------------
    # BaseSurrogate 추상 메서드 구현
    # ------------------------------------------------------------------

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """학습 데이터로 Kriging 서로게이트를 학습한다.

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
            "KrigingSurrogate.fit: n_samples=%d, n_features=%d, n_outputs=%d",
            n_samples,
            n_features,
            n_outputs,
        )

        if self._try_fit_smt(X, y_2d):
            self._backend = "smt"
        elif self._try_fit_sklearn_gp(X, y_2d):
            self._backend = "sklearn_gp"
        else:
            # _fit_linear_fallback 내부에서 _backend를 설정함
            self._fit_linear_fallback(X, y_2d)

        self.is_fitted = True
        logger.info("KrigingSurrogate 학습 완료 (backend=%s).", self._backend)

    def _try_fit_smt(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> bool:
        """SMT Kriging으로 학습을 시도한다.

        Args:
            X: 입력 데이터.
            y: 출력 데이터. shape = (n_samples, n_outputs).

        Returns:
            성공 시 True, 실패 시 False.
        """
        try:
            from smt.surrogate_models import KRG  # type: ignore[import]

            models = []
            for i in range(y.shape[1]):
                sm = KRG(
                    corr=self.corr,
                    poly=self.poly,
                    print_global=False,
                )
                sm.set_training_values(X, y[:, i])
                sm.train()
                models.append(sm)
            self._model = models
            return True
        except ImportError:
            logger.warning("smt 미설치 — 대체 백엔드를 시도합니다.")
            return False
        except Exception as exc:
            logger.warning("SMT Kriging 학습 실패: %s — 대체 백엔드를 시도합니다.", exc)
            return False

    def _try_fit_sklearn_gp(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> bool:
        """sklearn GaussianProcessRegressor로 학습을 시도한다.

        Args:
            X: 입력 데이터.
            y: 출력 데이터. shape = (n_samples, n_outputs).

        Returns:
            성공 시 True, 실패 시 False.
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore[import]
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel  # type: ignore[import]

            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            models = []
            for i in range(y.shape[1]):
                gpr = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=3,
                    normalize_y=True,
                    random_state=42,
                )
                gpr.fit(X, y[:, i])
                models.append(gpr)
            self._model = models
            return True
        except ImportError:
            logger.warning("sklearn 미설치 — LinearRegression으로 폴백합니다.")
            return False
        except Exception as exc:
            logger.warning("sklearn GP 학습 실패: %s — LinearRegression으로 폴백합니다.", exc)
            return False

    def _fit_linear_fallback(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> None:
        """sklearn LinearRegression 또는 numpy lstsq 최종 폴백 학습.

        Args:
            X: 입력 데이터.
            y: 출력 데이터. shape = (n_samples, n_outputs).
        """
        try:
            from sklearn.linear_model import LinearRegression  # type: ignore[import]

            model = LinearRegression()
            model.fit(X, y)
            self._model = [model]
            self._linear_model = model
            self._backend = "sklearn_linear"
            return
        except ImportError:
            pass

        # numpy lstsq 최종 폴백 (외부 의존성 없음)
        logger.warning("sklearn 미설치 — numpy lstsq 최소 제곱 폴백을 사용합니다.")
        X_aug = np.column_stack([X, np.ones(X.shape[0])])
        coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        self._model = None
        self._numpy_coeffs = coeffs   # (n_features+1, n_outputs)
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
        elif self._backend == "sklearn_gp":
            preds = np.column_stack(
                [gpr.predict(X) for gpr in self._model]
            )
        elif self._backend == "sklearn_linear":
            preds = self._linear_model.predict(X)
            if preds.ndim == 1:
                preds = preds[:, np.newaxis]
        else:
            # numpy lstsq 폴백
            X_aug = np.column_stack([X, np.ones(X.shape[0])])
            preds = X_aug @ self._numpy_coeffs

        if preds.ndim == 1:
            preds = preds[:, np.newaxis]

        if preds.shape[1] == 1:
            return preds.ravel()
        return preds

    def predict_with_variance(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """예측값과 예측 분산을 함께 반환한다.

        SMT 백엔드는 predict_variances()를 사용하고,
        sklearn GP 백엔드는 return_std=True를 사용한다.
        폴백(LinearRegression) 백엔드는 분산을 0으로 반환한다.

        Args:
            X: 예측할 입력. shape = (n_samples, n_features).

        Returns:
            (y_pred, y_var) 튜플.
                y_pred: 예측값. shape = (n_samples,) 또는 (n_samples, n_outputs).
                y_var: 예측 분산. shape = (n_samples,) 또는 (n_samples, n_outputs).

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
            variances = np.column_stack(
                [np.maximum(sm.predict_variances(X).ravel(), 0.0) for sm in self._model]
            )
        elif self._backend == "sklearn_gp":
            results = [gpr.predict(X, return_std=True) for gpr in self._model]
            preds = np.column_stack([r[0] for r in results])
            variances = np.column_stack([r[1] ** 2 for r in results])
        elif self._backend == "sklearn_linear":
            # LinearRegression 폴백 — 분산을 0으로 반환
            preds_raw = self._linear_model.predict(X)
            if preds_raw.ndim == 1:
                preds_raw = preds_raw[:, np.newaxis]
            preds = preds_raw
            variances = np.zeros_like(preds)
        else:
            # numpy lstsq 폴백 — 분산을 0으로 반환
            X_aug = np.column_stack([X, np.ones(X.shape[0])])
            preds = X_aug @ self._numpy_coeffs
            variances = np.zeros_like(preds)

        # 단일 출력이면 1D로 반환
        if preds.shape[1] == 1:
            return preds.ravel(), variances.ravel()
        return preds, variances

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
            {"corr": ..., "poly": ..., "backend": ...}
        """
        return {"corr": self.corr, "poly": self.poly, "backend": self._backend}
