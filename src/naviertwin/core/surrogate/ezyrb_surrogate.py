"""EZyRB 기반 서로게이트 (POD-GPR / POD-NN).

mathLab `EZyRB <https://github.com/mathLab/EZyRB>`_ 의 GPR/ANN 근사기를
NavierTwin 의 :class:`~naviertwin.core.surrogate.base.BaseSurrogate` 계약
(``fit(X, y)`` / ``predict(X)``)에 맞춰 래핑한다. :class:`TwinEngine` 의
계수 회귀기로 쓰이면 각각 POD-GPR(불확실성 정량화 지원), POD-NN(신경망
회귀) 조합이 된다.

ezyrb 는 선택 의존성이다 — 미설치 환경에서도 모듈 import 는 성공하고,
``fit()`` 호출 시점에 명확한 RuntimeError 를 던진다 (pip install ezyrb).
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.surrogate.base import BaseSurrogate
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

_EZYRB_INSTALL_HINT = (
    "EZyRB 서로게이트에는 ezyrb 패키지가 필요합니다 — pip install ezyrb"
)


def _import_ezyrb_class(name: str) -> Any:
    """ezyrb 근사기 클래스를 lazy import 한다.

    Args:
        name: ezyrb 클래스 이름 (``"GPR"`` | ``"ANN"``).

    Returns:
        요청한 ezyrb 클래스.

    Raises:
        RuntimeError: ezyrb 가 설치되지 않은 경우.
    """
    try:
        import ezyrb
    except ImportError as exc:
        raise RuntimeError(_EZYRB_INSTALL_HINT) from exc
    return getattr(ezyrb, name)


class EzyRBGPRSurrogate(BaseSurrogate):
    """EZyRB GPR (Gaussian Process Regression) 서로게이트.

    sklearn ``GaussianProcessRegressor`` 기반(ezyrb.GPR)이라 예측
    표준편차(UQ)를 함께 제공한다. TwinEngine 의 계수 회귀기로 쓰면
    POD-GPR 조합이 된다.

    참고: ezyrb ``GPR.predict(..., return_variance=True)`` 는 이름과 달리
    sklearn 의 ``return_std=True`` 결과(표준편차)를 그대로 돌려준다 —
    이 래퍼는 이를 표준편차로 취급하고 :meth:`predict_with_variance` 에서만
    분산(std²)으로 변환한다.

    Attributes:
        optimization_restart: 커널 하이퍼파라미터 최적화 재시작 횟수.
        last_std_: 가장 최근 예측의 표준편차. shape = (n_samples, n_outputs).
            fit 직후에는 학습점에서의 std 로 초기화된다 (보간점이므로 ~0).

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.surrogate.ezyrb_surrogate import EzyRBGPRSurrogate
        >>> rng = np.random.default_rng(0)
        >>> X = rng.uniform(-1, 1, (20, 2))
        >>> y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        >>> gpr = EzyRBGPRSurrogate()
        >>> gpr.fit(X, y)
        >>> y_pred = gpr.predict(X)
    """

    def __init__(self, optimization_restart: int = 10) -> None:
        """초기화.

        Args:
            optimization_restart: 커널 최적화 재시작 횟수 (기본: 10).
                ezyrb 기본값(20)보다 줄여 소량 샘플 학습을 빠르게 한다.
        """
        super().__init__()
        self.optimization_restart = optimization_restart
        self._model: Any = None
        self.last_std_: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # BaseSurrogate 추상 메서드 구현
    # ------------------------------------------------------------------

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """학습 데이터로 GPR 서로게이트를 학습한다.

        Args:
            X: 입력 설계 변수. shape = (n_samples, n_features).
            y: 출력 응답값. shape = (n_samples,) 또는 (n_samples, n_outputs).

        Raises:
            ValueError: 입력 shape 이 올바르지 않은 경우.
            RuntimeError: ezyrb 가 설치되지 않은 경우.
        """
        if X.ndim != 2:
            raise ValueError(f"X는 2D 배열이어야 합니다. 현재 shape: {X.shape}")

        gpr_cls = _import_ezyrb_class("GPR")

        X = np.asarray(X, dtype=np.float64)
        y_2d = np.asarray(y if y.ndim == 2 else y[:, np.newaxis], dtype=np.float64)
        self.input_dim = int(X.shape[1])
        self.output_dim = int(y_2d.shape[1])

        logger.debug(
            "EzyRBGPRSurrogate.fit: n_samples=%d, n_features=%d, n_outputs=%d",
            X.shape[0],
            self.input_dim,
            self.output_dim,
        )

        model = gpr_cls(optimization_restart=self.optimization_restart)
        model.fit(X, y_2d)
        self._model = model
        self.is_fitted = True

        # 학습점에서의 예측 std 로 last_std_ 초기화 — TwinEngine 학습 직후
        # 메타데이터(uq_mean_std)가 참조한다. UQ 실패가 fit 을 막지는 않는다.
        try:
            self._predict_with_std(X)
        except Exception as exc:  # noqa: BLE001 — UQ 는 부가 기능
            logger.warning("GPR 학습점 std 계산 실패 (무시): %s", exc)
        logger.info("EzyRBGPRSurrogate 학습 완료 (ezyrb GPR).")

    def _predict_with_std(
        self, X_2d: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """평균과 표준편차를 (n_samples, n_outputs) 로 정규화해 반환한다.

        sklearn 버전에 따라 std 가 (n_samples,) 또는 (n_samples, n_outputs)
        로 나오므로 항상 2D 로 브로드캐스트하고 ``last_std_`` 를 갱신한다.

        Args:
            X_2d: 예측할 입력. shape = (n_samples, n_features).

        Returns:
            (mean, std) — 둘 다 shape = (n_samples, n_outputs).
        """
        mean, std = self._model.predict(X_2d, return_variance=True)
        mean = np.asarray(mean, dtype=np.float64)
        if mean.ndim == 1:
            mean = mean[:, np.newaxis]
        std = np.asarray(std, dtype=np.float64)
        if std.ndim == 1:
            std = np.repeat(std[:, np.newaxis], mean.shape[1], axis=1)
        self.last_std_ = std
        return mean, std

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """새로운 입력에 대해 출력을 예측한다.

        예측 표준편차는 ``last_std_`` 속성에 함께 갱신된다.

        Args:
            X: 예측할 입력. shape = (n_samples, n_features).

        Returns:
            예측값. shape = (n_samples, n_outputs) 또는 (n_samples,).

        Raises:
            RuntimeError: fit() 이 호출되지 않은 경우.
        """
        self._check_fitted()

        if X.ndim == 1:
            X = X[np.newaxis, :]
        preds, _ = self._predict_with_std(np.asarray(X, dtype=np.float64))

        if preds.shape[1] == 1:
            return preds.ravel()
        return preds

    def predict_with_variance(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """예측값과 예측 분산(std²)을 함께 반환한다.

        :class:`KrigingSurrogate.predict_with_variance` 와 동일한 계약.

        Args:
            X: 예측할 입력. shape = (n_samples, n_features).

        Returns:
            (y_pred, y_var) 튜플. 단일 출력이면 각각 1D 로 반환한다.

        Raises:
            RuntimeError: fit() 이 호출되지 않은 경우.
        """
        self._check_fitted()

        if X.ndim == 1:
            X = X[np.newaxis, :]
        preds, std = self._predict_with_std(np.asarray(X, dtype=np.float64))
        variances = std**2

        if preds.shape[1] == 1:
            return preds.ravel(), variances.ravel()
        return preds, variances

    def get_params(self) -> dict[str, Any]:
        """하이퍼파라미터를 반환한다.

        Returns:
            {"optimization_restart": ..., "backend": "ezyrb_gpr"}
        """
        return {
            "optimization_restart": self.optimization_restart,
            "backend": "ezyrb_gpr",
        }


class EzyRBANNSurrogate(BaseSurrogate):
    """EZyRB ANN (torch MLP) 서로게이트 — POD-NN 계수 회귀기.

    입력/출력을 내부에서 표준화한 뒤 ezyrb.ANN(작은 tanh MLP)으로 학습한다
    — POD 계수는 스케일이 커서 원시 MSE 학습이 잘 수렴하지 않기 때문.
    ezyrb ANN 이 epoch 마다 print 하는 학습 로그는 stdout 리다이렉트로
    흡수한다.

    Attributes:
        layers: 은닉층 폭 튜플.
        max_epochs: 최대 학습 epoch 수 (stop_training 예산).
        tolerance: 학습 손실 조기 종료 허용치.
        lr: Adam 학습률.
        seed: torch 시드 (재현성).

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.surrogate.ezyrb_surrogate import EzyRBANNSurrogate
        >>> rng = np.random.default_rng(0)
        >>> X = rng.uniform(-1, 1, (20, 2))
        >>> y = np.column_stack([np.sin(X[:, 0]), X[:, 1] ** 2])
        >>> ann = EzyRBANNSurrogate(layers=(8, 8), max_epochs=200)
        >>> ann.fit(X, y)
        >>> y_pred = ann.predict(X)
    """

    def __init__(
        self,
        layers: tuple[int, ...] = (16, 16),
        max_epochs: int = 2000,
        tolerance: float = 1e-6,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        """초기화.

        Args:
            layers: 은닉층 폭 (기본: (16, 16)).
            max_epochs: 최대 epoch 수 (기본: 2000).
            tolerance: 손실이 이 값 아래로 내려가면 조기 종료 (기본: 1e-6).
            lr: Adam 학습률 (기본: 1e-3).
            seed: torch 난수 시드 (기본: 0).
        """
        super().__init__()
        self.layers = tuple(int(w) for w in layers)
        self.max_epochs = int(max_epochs)
        self.tolerance = float(tolerance)
        self.lr = float(lr)
        self.seed = int(seed)
        self._model: Any = None
        self._x_mean: NDArray[np.float64] | None = None
        self._x_scale: NDArray[np.float64] | None = None
        self._y_mean: NDArray[np.float64] | None = None
        self._y_scale: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # BaseSurrogate 추상 메서드 구현
    # ------------------------------------------------------------------

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """학습 데이터로 ANN 서로게이트를 학습한다.

        Args:
            X: 입력 설계 변수. shape = (n_samples, n_features).
            y: 출력 응답값. shape = (n_samples,) 또는 (n_samples, n_outputs).

        Raises:
            ValueError: 입력 shape 이 올바르지 않은 경우.
            RuntimeError: ezyrb 가 설치되지 않은 경우.
        """
        if X.ndim != 2:
            raise ValueError(f"X는 2D 배열이어야 합니다. 현재 shape: {X.shape}")

        ann_cls = _import_ezyrb_class("ANN")
        import torch
        from torch import nn

        X = np.asarray(X, dtype=np.float64)
        y_2d = np.asarray(y if y.ndim == 2 else y[:, np.newaxis], dtype=np.float64)
        self.input_dim = int(X.shape[1])
        self.output_dim = int(y_2d.shape[1])

        logger.debug(
            "EzyRBANNSurrogate.fit: n_samples=%d, n_features=%d, n_outputs=%d, "
            "layers=%s, max_epochs=%d",
            X.shape[0],
            self.input_dim,
            self.output_dim,
            self.layers,
            self.max_epochs,
        )

        # 입출력 표준화 (0-분산 성분은 scale=1 로 보호).
        self._x_mean = X.mean(axis=0)
        self._x_scale = np.where(X.std(axis=0) > 0.0, X.std(axis=0), 1.0)
        self._y_mean = y_2d.mean(axis=0)
        self._y_scale = np.where(y_2d.std(axis=0) > 0.0, y_2d.std(axis=0), 1.0)
        X_scaled = (X - self._x_mean) / self._x_scale
        y_scaled = (y_2d - self._y_mean) / self._y_scale

        torch.manual_seed(self.seed)
        model = ann_cls(
            list(self.layers),
            nn.Tanh(),
            [self.max_epochs, self.tolerance],
            lr=self.lr,
            frequency_print=10**9,  # 주기 출력 억제 (첫/마지막 epoch 은 아래에서 흡수)
        )
        # ezyrb ANN.fit 은 print() 로 손실을 찍는다 — 웹/CLI 로그 오염 방지.
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X_scaled, y_scaled)
        self._model = model

        self.is_fitted = True
        logger.info(
            "EzyRBANNSurrogate 학습 완료 (epochs<=%d, final_loss=%s).",
            self.max_epochs,
            model.loss_trend[-1] if getattr(model, "loss_trend", None) else "?",
        )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """새로운 입력에 대해 출력을 예측한다.

        Args:
            X: 예측할 입력. shape = (n_samples, n_features).

        Returns:
            예측값. shape = (n_samples, n_outputs) 또는 (n_samples,).

        Raises:
            RuntimeError: fit() 이 호출되지 않은 경우.
        """
        self._check_fitted()

        if X.ndim == 1:
            X = X[np.newaxis, :]
        X_scaled = (np.asarray(X, dtype=np.float64) - self._x_mean) / self._x_scale

        preds_scaled = np.asarray(self._model.predict(X_scaled), dtype=np.float64)
        if preds_scaled.ndim == 1:
            preds_scaled = preds_scaled[np.newaxis, :]
        preds = preds_scaled * self._y_scale + self._y_mean

        if preds.shape[1] == 1:
            return preds.ravel()
        return preds

    def get_params(self) -> dict[str, Any]:
        """하이퍼파라미터를 반환한다.

        Returns:
            {"layers": ..., "max_epochs": ..., "tolerance": ..., "lr": ...,
            "seed": ..., "backend": "ezyrb_ann"}
        """
        return {
            "layers": self.layers,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
            "lr": self.lr,
            "seed": self.seed,
            "backend": "ezyrb_ann",
        }
