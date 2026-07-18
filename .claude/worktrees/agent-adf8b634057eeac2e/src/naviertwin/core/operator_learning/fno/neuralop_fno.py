"""neuraloperator(레퍼런스 구현) FNO 래퍼.

`neuraloperator <https://github.com/neuraloperator/neuraloperator>`_ 는 FNO 논문
저자들이 유지하는 표준 구현이다(PyTorch Ecosystem). 이 앱의 자체 FNO
(:mod:`naviertwin.core.operator_learning.fno.fno`)와 **같은 fit/predict 계약**
으로 감싸 서로 바꿔 끼울 수 있게 한다 — 같은 벤치마크에서 직접 비교 가능.

두 가지를 흡수한다:
    - **텐서 레이아웃**: neuralop 은 채널 우선 ``(B, C, ...)``, 이 앱의 데이터셋은
      채널 마지막 ``(B, ..., C)`` 다. 래퍼가 transpose 를 담당한다.
    - **학습 루프**: neuralop 의 ``FNO`` 는 순수 ``nn.Module`` 이라 fit 이 없다.
      자체 FNO 와 동일한 Adam + MSE 루프를 여기서 돌린다(``epoch_callback`` 으로
      라이브 진행 스트리밍도 동일하게 지원).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def neuralop_available() -> bool:
    """neuraloperator 패키지 사용 가능 여부."""
    try:
        import neuralop  # noqa: F401
    except ImportError:
        return False
    return True


class NeuralOpFNO(BaseOperator):
    """neuraloperator 의 FNO 를 앱의 연산자 계약으로 감싼 래퍼.

    1D/2D 를 ``n_dim`` 으로 함께 지원한다 — neuralop 의 ``FNO`` 는 ``n_modes``
    튜플 길이로 차원이 정해지기 때문에 클래스를 나눌 필요가 없다.

    Attributes:
        n_dim: 공간 차원 (1 또는 2).
        train_losses_: epoch 별 평균 손실 (자체 FNO 와 동일한 계약).

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.operator_learning.fno.neuralop_fno import NeuralOpFNO
        >>> op = NeuralOpFNO(in_channels=1, out_channels=1, modes=4, width=8,
        ...                  n_dim=1, max_epochs=1)
        >>> x = np.random.rand(4, 16, 1)
        >>> op.fit({"inputs": x, "outputs": x})       # doctest: +SKIP
        >>> op.predict({"x": x}).shape                # doctest: +SKIP
        (4, 16, 1)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: int | Sequence[int] = 12,
        width: int = 32,
        n_dim: int = 1,
        n_layers: int = 4,
        max_epochs: int = 60,
        batch_size: int = 16,
        lr: float = 1e-3,
        seed: int | None = 0,
        epoch_callback: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        """초기화.

        Args:
            in_channels: 입력 채널 수.
            out_channels: 출력 채널 수.
            modes: 유지할 푸리에 모드 수. 정수면 전 축에 동일 적용.
            width: 은닉 채널 폭 (neuralop 의 ``hidden_channels``).
            n_dim: 공간 차원 (1 | 2).
            n_layers: 푸리에 층 수.
            max_epochs: 학습 epoch 수.
            batch_size: 미니배치 크기.
            lr: Adam 학습률.
            seed: 난수 시드.
            epoch_callback: ``(epoch_idx, epoch_loss)`` 라이브 진행 콜백.

        Raises:
            ValueError: ``n_dim`` 이 1/2 가 아닌 경우.
        """
        super().__init__()
        if n_dim not in (1, 2):
            raise ValueError(f"n_dim 은 1 또는 2 여야 합니다. 현재: {n_dim}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_dim = int(n_dim)
        self.modes = modes
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.seed = seed
        self.epoch_callback = epoch_callback
        self.train_losses_: list[float] = []
        self._model: Any = None
        self._device: Any = None

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _modes_tuple(self) -> tuple[int, ...]:
        if isinstance(self.modes, (int, np.integer)):
            return tuple([int(self.modes)] * self.n_dim)
        values = tuple(int(m) for m in self.modes)
        if len(values) != self.n_dim:
            raise ValueError(
                f"modes 길이({len(values)})가 n_dim({self.n_dim})과 다릅니다."
            )
        return values

    def _build(self) -> Any:
        from neuralop.models import FNO

        return FNO(
            n_modes=self._modes_tuple(),
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            hidden_channels=self.width,
            n_layers=self.n_layers,
        )

    @staticmethod
    def _to_channel_first(arr: NDArray[np.float64]) -> NDArray[np.float32]:
        """(B, ..., C) → (B, C, ...) — neuralop 은 채널 우선."""
        values = np.asarray(arr, dtype=np.float32)
        axes = (0, values.ndim - 1, *range(1, values.ndim - 1))
        return np.transpose(values, axes)

    @staticmethod
    def _to_channel_last(arr: NDArray[np.float32]) -> NDArray[np.float64]:
        """(B, C, ...) → (B, ..., C) — 앱 데이터셋 규약으로 복귀."""
        values = np.asarray(arr)
        axes = (0, *range(2, values.ndim), 1)
        return np.transpose(values, axes).astype(np.float64, copy=False)

    def _resolve_device(self) -> Any:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 학습 / 추론
    # ------------------------------------------------------------------

    def fit(self, dataset: dict[str, Any]) -> None:
        """연산자를 학습한다 (Adam + MSE — 자체 FNO 와 동일 조건).

        Args:
            dataset: ``inputs`` (B, ..., C_in), ``outputs`` (B, ..., C_out).

        Raises:
            ImportError: neuraloperator/torch 미설치.
            ValueError: 입출력 shape 이 ``n_dim`` 과 맞지 않는 경우.
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise ImportError("torch 가 필요합니다: pip install torch") from exc
        if not neuralop_available():
            raise ImportError(
                "neuraloperator 가 설치되어 있지 않습니다. "
                "`pip install neuraloperator` 로 설치하세요."
            )

        X = np.asarray(dataset["inputs"], dtype=np.float64)
        Y = np.asarray(dataset["outputs"], dtype=np.float64)
        expected = self.n_dim + 2  # (B, *spatial, C)
        if X.ndim != expected or Y.ndim != expected:
            raise ValueError(
                f"n_dim={self.n_dim} 은 {expected}D (B, *spatial, C) 입력이 "
                f"필요합니다. 현재: inputs={X.shape}, outputs={Y.shape}"
            )

        if self.seed is not None:
            torch.manual_seed(int(self.seed))
        self._device = self._resolve_device()
        self._model = self._build().to(self._device)

        tx = torch.tensor(self._to_channel_first(X), device=self._device)
        ty = torch.tensor(self._to_channel_first(Y), device=self._device)
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        n_samples = tx.shape[0]
        batch = max(1, min(self.batch_size, n_samples))

        self.train_losses_ = []
        self._model.train()
        for epoch_idx in range(self.max_epochs):
            perm = torch.randperm(n_samples, device=self._device)
            total, seen = 0.0, 0
            for start in range(0, n_samples, batch):
                idx = perm[start : start + batch]
                optim.zero_grad()
                pred = self._model(tx[idx])
                loss = loss_fn(pred, ty[idx])
                loss.backward()
                optim.step()
                total += float(loss.item()) * int(idx.numel())
                seen += int(idx.numel())
            epoch_loss = total / max(1, seen)
            self.train_losses_.append(epoch_loss)
            if self.epoch_callback is not None:
                self.epoch_callback(epoch_idx, epoch_loss)

        self.is_fitted = True
        logger.info(
            "NeuralOpFNO 학습 완료: n_dim=%d, modes=%s, width=%d, final_loss=%.4g",
            self.n_dim,
            self._modes_tuple(),
            self.width,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, inputs: dict[str, Any]) -> NDArray[np.float64]:
        """학습된 연산자로 추론한다.

        배치 차원은 있어도 없어도 된다 — 자체 FNO 와 같은 관용성을 유지한다
        (``bench.evaluate_sample`` 은 단일 샘플을 배치 없이 넘긴다). 배치 없이
        들어오면 결과도 배치 없이 돌려준다.

        Args:
            inputs: ``x`` — (B, ..., C_in) 또는 (..., C_in).

        Returns:
            입력과 같은 배치 유무·채널 마지막 레이아웃의 예측.
        """
        import torch

        self._check_fitted()
        X = np.asarray(inputs["x"], dtype=np.float64)
        batched = X.ndim == self.n_dim + 2
        if not batched:
            if X.ndim != self.n_dim + 1:
                raise ValueError(
                    f"n_dim={self.n_dim} 은 (B, *spatial, C) 또는 (*spatial, C) "
                    f"입력이 필요합니다. 현재: {X.shape}"
                )
            X = X[np.newaxis, ...]

        tx = torch.tensor(self._to_channel_first(X), device=self._device)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(tx)
        result = self._to_channel_last(pred.cpu().numpy())
        return result if batched else result[0]


__all__ = ["NeuralOpFNO", "neuralop_available"]
