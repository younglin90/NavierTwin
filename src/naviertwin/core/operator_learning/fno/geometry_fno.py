"""형상 인지(geometry-aware) FNO — SDF/마스크/운전조건 채널 입력.

형상이 케이스마다 다른 정상(steady) 케이스 세트를 **공통 균일 격자** 위에서
학습하는 서로게이트. 형상은 부호거리(SDF)·유체 마스크 채널로, 운전조건은 상수
브로드캐스트 채널로 입력에 들어간다 — DeepCFD(Ribeiro et al., 2020)·Thuerey
et al.(2020) 계열의 표준 인코딩이다. 샘플 하나 = 케이스 하나(시간축 없음).

입력 텐서는 :func:`naviertwin.core.operator_learning.fno.case_tensorizer.
cases_to_grid_tensors` 가 만든다: 채널 = ``[sdf, mask, μ_1, ..., μ_k]``.

주의(소표본): 문헌의 데이터 규모는 수백~수천 케이스다(DeepCFD ~1000개 형상).
이 클래스는 5~10 케이스의 소표본에서도 **동작은 하도록**(크래시 없이 학습·추론)
설계했지만, 그 결과는 정량 예측이 아니라 정성적 데모로 해석해야 한다.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.fno.geometry_fno import GeometryFNO2D
    >>> N, H, W = 5, 24, 32
    >>> X = np.random.rand(N, H, W, 3).astype(np.float32)  # [sdf, mask, mu_0]
    >>> Y = np.random.rand(N, H, W, 1).astype(np.float32)  # [p]
    >>> op = GeometryFNO2D(n_params=1, modes=6, width=8, epochs=2)
    >>> op.fit(X, Y)                              # doctest: +SKIP
    >>> op.predict(X).shape                       # doctest: +SKIP
    (5, 24, 32, 1)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

#: 지원하는 FNO 백엔드 — "builtin"(자체 FNO2D) | "neuralop"(레퍼런스 구현).
GEOMETRY_FNO_BACKENDS = ("builtin", "neuralop")


class GeometryFNO2D:
    """SDF/마스크/운전조건 채널을 입력받는 2D 형상 인지 FNO 서로게이트.

    기존 FNO 기계(:class:`~naviertwin.core.operator_learning.fno.fno.FNO2D`
    또는 :class:`~naviertwin.core.operator_learning.fno.neuralop_fno.
    NeuralOpFNO`)를 그대로 재사용하고, 이 클래스는 그 위에서

    - 채널 계약 검증 (입력 = 2 + n_params 채널, 출력 = out_channels 채널),
    - 채널별 표준화/역표준화 (평균·표준편차 저장),
    - ndarray 직접 입출력 인터페이스 (``fit(inputs, targets)``)

    만 담당한다. ``fit(dict)`` 계약의 :class:`BaseOperator` 와 시그니처가
    달라(배열 2개 직접 전달) 상속 대신 합성(composition)을 쓴다.

    Attributes:
        n_params: 운전조건 파라미터 수 (입력 채널 = 2 + n_params).
        in_channels: 입력 채널 수 (= 2 + n_params).
        out_channels: 출력(타깃 필드) 채널 수.
        backend: 사용 중인 FNO 백엔드 ("builtin" | "neuralop").
        train_losses_: epoch 별 평균 학습 손실 (표준화 공간의 MSE).
        is_fitted: :meth:`fit` 완료 여부.
        input_mean_ / input_std_: 입력 채널별 표준화 통계. shape = (C_in,).
        output_mean_ / output_std_: 출력 채널별 표준화 통계. shape = (C_out,).
    """

    def __init__(
        self,
        n_params: int,
        out_channels: int = 1,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        epochs: int = 200,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
        backend: str = "builtin",
        epoch_callback: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        """초기화.

        Args:
            n_params: 운전조건 파라미터 수 (k ≥ 0). 입력 채널 = 2 + k.
            out_channels: 출력 필드 채널 수.
            modes: 축별 유지 푸리에 모드 수 (두 축 동일 적용).
            width: 은닉 채널 폭.
            n_layers: 푸리에 층 수.
            epochs: 학습 epoch 수 (Adam + MSE).
            lr: Adam 학습률.
            device: "auto" | "cpu" | "cuda" 등. "neuralop" 백엔드는 자체
                규칙(cuda 가능 시 cuda)을 따르므로 이 값이 무시된다.
            seed: 난수 시드. None 이면 고정하지 않는다.
            backend: "builtin"(자체 FNO2D) | "neuralop"(레퍼런스 FNO).
            epoch_callback: ``(epoch_idx, epoch_loss)`` 라이브 진행 콜백.

        Raises:
            ValueError: ``n_params`` 가 음수이거나 ``backend`` 가 지원 목록에
                없는 경우.
        """
        if int(n_params) < 0:
            raise ValueError(f"n_params 는 0 이상이어야 합니다. 현재: {n_params}")
        if backend not in GEOMETRY_FNO_BACKENDS:
            raise ValueError(
                f"지원하지 않는 backend: {backend!r}. 지원: {list(GEOMETRY_FNO_BACKENDS)}"
            )
        self.n_params = int(n_params)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = device
        self.seed = seed
        self.backend = backend
        self.epoch_callback = epoch_callback

        self.is_fitted: bool = False
        self.train_losses_: list[float] = []
        self.input_mean_: NDArray[np.float32] | None = None
        self.input_std_: NDArray[np.float32] | None = None
        self.output_mean_: NDArray[np.float32] | None = None
        self.output_std_: NDArray[np.float32] | None = None
        self._model: Any = None

    @property
    def in_channels(self) -> int:
        """입력 채널 수 (= sdf + mask + 파라미터 k개)."""
        return 2 + self.n_params

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    @staticmethod
    def _channel_stats(
        values: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """(N, H, W, C) 텐서의 채널별 평균/표준편차 (상수 채널은 std=1)."""
        mean = values.mean(axis=(0, 1, 2)).astype(np.float32)
        std = values.std(axis=(0, 1, 2)).astype(np.float32)
        std = np.where(std < 1e-8, np.float32(1.0), std)
        return mean, std

    def _build_model(self) -> Any:
        """설정에 맞는 백엔드 FNO 인스턴스를 만든다."""
        if self.backend == "neuralop":
            from naviertwin.core.operator_learning.fno.neuralop_fno import NeuralOpFNO

            return NeuralOpFNO(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                modes=self.modes,
                width=self.width,
                n_dim=2,
                n_layers=self.n_layers,
                max_epochs=self.epochs,
                lr=self.lr,
                seed=self.seed,
                epoch_callback=self.epoch_callback,
            )
        from naviertwin.core.operator_learning.fno.fno import FNO2D

        return FNO2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            modes1=self.modes,
            modes2=self.modes,
            width=self.width,
            n_layers=self.n_layers,
            max_epochs=self.epochs,
            lr=self.lr,
            device=self.device,
            seed=self.seed,
            epoch_callback=self.epoch_callback,
        )

    def _validate_inputs(self, values: NDArray[np.float32], what: str) -> None:
        if values.ndim != 4:
            raise ValueError(
                f"{what} 는 (N, H, W, C) 4D 텐서여야 합니다. 현재: {values.shape}"
            )

    # ------------------------------------------------------------------
    # 학습 / 추론
    # ------------------------------------------------------------------

    def fit(
        self,
        inputs: NDArray[np.float32],
        targets: NDArray[np.float32],
    ) -> None:
        """케이스 텐서로 형상 인지 FNO 를 학습한다 (Adam + MSE).

        입출력을 채널별로 표준화한 뒤 백엔드 FNO 에 위임한다. 소표본
        (N = 5~10)에서도 동작하지만 결과는 정성적 데모 수준이다 — 문헌
        (DeepCFD 등)의 정량 성능은 수백~수천 케이스에서 얻어진다.

        Args:
            inputs: (N, H, W, 2 + n_params) float32 — [sdf, mask, μ...].
            targets: (N, H, W, out_channels) float32 — 타깃 필드 채널.

        Raises:
            ValueError: shape/채널 수가 계약과 맞지 않는 경우.
        """
        x = np.asarray(inputs, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        self._validate_inputs(x, "inputs")
        self._validate_inputs(y, "targets")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"입력 채널 수가 계약(2 + n_params = {self.in_channels})과 "
                f"다릅니다. 현재: {x.shape[-1]} (shape={x.shape})"
            )
        if y.shape[-1] != self.out_channels:
            raise ValueError(
                f"타깃 채널 수가 out_channels({self.out_channels})와 다릅니다. "
                f"현재: {y.shape[-1]} (shape={y.shape})"
            )
        if x.shape[:3] != y.shape[:3]:
            raise ValueError(
                f"inputs/targets 의 (N, H, W) 가 다릅니다: {x.shape[:3]} vs {y.shape[:3]}"
            )
        if x.shape[0] < 1:
            raise ValueError("학습 케이스가 최소 1개 필요합니다.")

        self.input_mean_, self.input_std_ = self._channel_stats(x)
        self.output_mean_, self.output_std_ = self._channel_stats(y)
        x_std = (x - self.input_mean_) / self.input_std_
        y_std = (y - self.output_mean_) / self.output_std_

        self._model = self._build_model()
        self._model.fit({"inputs": x_std, "outputs": y_std})
        self.train_losses_ = list(self._model.train_losses_)
        self.is_fitted = True
        logger.info(
            "GeometryFNO2D 학습 완료: backend=%s, N=%d, in=%d ch, out=%d ch, loss=%.6g",
            self.backend,
            x.shape[0],
            self.in_channels,
            self.out_channels,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """새 형상/운전조건 텐서에 대한 필드를 예측한다.

        Args:
            inputs: (N, H, W, 2 + n_params) 또는 (H, W, 2 + n_params).

        Returns:
            (N, H, W, out_channels) float32. 배치 없는 입력이면 배치 없는
            (H, W, out_channels) 를 돌려준다. 값은 원 물리 스케일로
            역표준화되어 있다.

        Raises:
            RuntimeError: :meth:`fit` 이 호출되지 않은 경우.
            ValueError: 입력 shape/채널 수가 계약과 맞지 않는 경우.
        """
        if not self.is_fitted or self._model is None:
            raise RuntimeError("GeometryFNO2D 의 fit() 을 먼저 호출해야 합니다.")
        x = np.asarray(inputs, dtype=np.float32)
        squeeze = x.ndim == 3
        if squeeze:
            x = x[np.newaxis, ...]
        self._validate_inputs(x, "inputs")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"입력 채널 수가 계약(2 + n_params = {self.in_channels})과 "
                f"다릅니다. 현재: {x.shape[-1]} (shape={x.shape})"
            )
        assert self.input_mean_ is not None and self.input_std_ is not None
        assert self.output_mean_ is not None and self.output_std_ is not None

        x_std = (x - self.input_mean_) / self.input_std_
        y_std = np.asarray(self._model.predict({"x": x_std}), dtype=np.float32)
        y = y_std * self.output_std_ + self.output_mean_
        return y[0] if squeeze else y

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"GeometryFNO2D(backend={self.backend!r}, n_params={self.n_params}, "
            f"out_channels={self.out_channels}, status={status})"
        )


__all__ = ["GEOMETRY_FNO_BACKENDS", "GeometryFNO2D"]
