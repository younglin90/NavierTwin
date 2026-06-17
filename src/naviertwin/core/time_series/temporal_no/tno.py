"""TNO — Temporal Neural Operator (경량 구현).

FNO 인코더로 공간 주파수 특징 추출 + 시간 branch (MLP 또는 GRU) 로 시간 의존성.
시간 외삽 오차 누적이 거의 없도록 직접 다중 스텝 예측.

References:
    Temporal Neural Operator, Nature Sci. Rep. 2025 (concept).

input 규약 (fit):
    - ``"sequences"``: (N, T, spatial, 1) 시계열 필드.
    - ``"dt"``: float (선택)
    - ``"horizon"``: 예측할 미래 스텝 수.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.time_series.temporal_no.tno import TNO
    >>> rng = np.random.default_rng(0)
    >>> seqs = rng.standard_normal((6, 12, 32, 1)).astype(np.float32)
    >>> m = TNO(spatial_size=32, channels=1, width=8, modes=4,
    ...         horizon=3, max_epochs=2)
    >>> m.fit(seqs)
    >>> m.predict(seqs[0, :4]).shape
    (3, 32, 1)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.fno.fno import _build_spectral_conv_1d
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class TNO:
    """FNO 인코더 + 시간 branch → horizon 스텝 동시 예측."""

    def __init__(
        self,
        spatial_size: int,
        channels: int = 1,
        width: int = 16,
        modes: int = 8,
        history: int = 4,
        horizon: int = 3,
        n_fno_layers: int = 2,
        max_epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        self.spatial_size = spatial_size
        self.channels = channels
        self.width = width
        self.modes = modes
        self.history = history
        self.horizon = horizon
        self.n_fno_layers = n_fno_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed

        self._model: Any = None
        self._device: Any = None
        self.is_fitted: bool = False
        self.train_losses_: list[float] = []

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self) -> Any:
        import torch
        import torch.nn as nn

        if self.seed is not None:
            torch.manual_seed(self.seed)

        W, M, C, H = self.width, self.modes, self.channels, self.history
        horizon = self.horizon
        n_layers = self.n_fno_layers

        class _TNO(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # history-step 입력: concat in channel 방향 (H * C)
                self.lift = nn.Linear(H * C, W)
                specs = nn.ModuleList()
                ws = nn.ModuleList()
                layer_idx = 0
                while layer_idx < n_layers:
                    specs.append(_build_spectral_conv_1d(W, W, M))
                    ws.append(nn.Conv1d(W, W, 1))
                    layer_idx += 1
                self.specs = specs
                self.ws = ws
                # 시간 branch — horizon * C 만큼 동시 예측
                self.proj = nn.Sequential(
                    nn.Linear(W, 4 * W), nn.GELU(),
                    nn.Linear(4 * W, horizon * C),
                )

            def forward(self, x: Any) -> Any:
                # x: (B, H, N, C) → (B, N, H*C)
                B, H_, N, C_ = x.shape
                x = x.permute(0, 2, 1, 3).reshape(B, N, H_ * C_)
                x = self.lift(x).permute(0, 2, 1)  # (B, W, N)
                layer_idx = 0
                while layer_idx < len(self.specs):
                    x = torch.nn.functional.gelu(
                        self.specs[layer_idx](x) + self.ws[layer_idx](x)
                    )
                    layer_idx += 1
                x = x.permute(0, 2, 1)  # (B, N, W)
                out = self.proj(x)  # (B, N, horizon * C)
                return out.reshape(B, N, horizon, C_).permute(0, 2, 1, 3)  # (B, horizon, N, C)

        return _TNO()

    def fit(self, sequences: np.ndarray) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seqs = np.asarray(sequences, dtype=np.float32)
        if seqs.ndim != 4:
            raise ValueError(f"sequences (N, T, spatial, C) 4D 필요: {seqs.shape}")
        N, T, S, C = seqs.shape
        if T < self.history + self.horizon:
            raise ValueError(
                f"T({T}) < history({self.history}) + horizon({self.horizon})"
            )
        if S != self.spatial_size or C != self.channels:
            raise ValueError("spatial_size / channels 불일치")

        n_windows = T - self.history - self.horizon + 1
        history_windows = np.lib.stride_tricks.sliding_window_view(
            seqs, self.history, axis=1
        )
        target_windows = np.lib.stride_tricks.sliding_window_view(
            seqs[:, self.history :, :, :], self.horizon, axis=1
        )
        X = np.moveaxis(history_windows[:, :n_windows], -1, 2).reshape(
            -1, self.history, S, C
        )
        Y = np.moveaxis(target_windows[:, :n_windows], -1, 2).reshape(
            -1, self.horizon, S, C
        )

        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        mse = torch.nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(Y)),
            batch_size=min(self.batch_size, len(X)),
            shuffle=True,
        )
        self.train_losses_ = []
        epoch_idx = 0
        while epoch_idx < self.max_epochs:
            epoch = 0.0
            batches = iter(loader)
            while True:
                try:
                    xb, yb = next(batches)
                except StopIteration:
                    break
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                pred = self._model(xb)
                loss = mse(pred, yb)
                loss.backward()
                optim.step()
                epoch += float(loss.item()) * xb.shape[0]
            epoch /= max(len(X), 1)
            self.train_losses_.append(epoch)
            epoch_idx += 1

        self.is_fitted = True
        logger.info("TNO 학습 완료: horizon=%d, loss=%.6g", self.horizon, self.train_losses_[-1])

    def predict(self, history_window: np.ndarray) -> np.ndarray:
        """(history, spatial, C) → (horizon, spatial, C)."""
        import torch

        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        x = np.asarray(history_window, dtype=np.float32)
        if x.ndim == 3:
            x = x[None, ...]
        with torch.no_grad():
            y = self._model(torch.tensor(x, device=self._device)).cpu().numpy()
        return y[0]


__all__ = ["TNO"]
