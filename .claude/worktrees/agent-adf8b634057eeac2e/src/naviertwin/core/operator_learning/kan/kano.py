"""KANO — Kolmogorov-Arnold Neural Operator (경량 구현).

KAN 의 핵심 아이디어: 가중치 대신 학습 가능한 단변량 활성화 함수.
여기서는 각 edge 에 학습 가능한 B-spline 기저 계수를 두는 간이 KAN 레이어를
FNO 블록의 pointwise 변환부에 삽입해 해석성을 보강한다.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.kan import KANO1D
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 32, 1)).astype(np.float32)
    >>> Y = np.sin(X).astype(np.float32)
    >>> op = KANO1D(in_channels=1, out_channels=1, modes=4, width=8,
    ...             grid_size=5, n_layers=2, max_epochs=3)
    >>> op.fit({"inputs": X, "outputs": Y})
    >>> op.predict({"x": X[:2]}).shape
    (2, 32, 1)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.core.operator_learning.fno.fno import _build_spectral_conv_1d
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _kan_layer(in_c: int, out_c: int, grid_size: int) -> Any:
    """단순 KAN layer — 각 입출력 쌍에 B-spline 계수.

    입력 x ∈ [-1, 1] 을 grid 에 샘플링하여 학습 가능한 스플라인 가중치를 곱한 뒤
    합산, bias linear 더함.
    """
    import torch
    import torch.nn as nn

    class _KAN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.grid = nn.Parameter(
                torch.linspace(-1.0, 1.0, grid_size), requires_grad=False
            )
            self.coef = nn.Parameter(
                0.01 * torch.randn(in_c, out_c, grid_size)
            )
            self.base = nn.Linear(in_c, out_c)

        def forward(self, x: Any) -> Any:
            # x: (..., in_c) — pointwise KAN
            orig_shape = x.shape
            x_flat = x.reshape(-1, in_c)
            # 그리드 거리 기반 RBF 가중치 (soft lookup)
            diff = x_flat.unsqueeze(-1) - self.grid  # (B, in_c, g)
            w = torch.exp(-4.0 * diff ** 2)  # (B, in_c, g)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
            # (B, in_c, g) × (in_c, out_c, g) → (B, out_c)
            out = torch.einsum("big,iog->bo", w, self.coef)
            out = out + self.base(x_flat)
            return out.reshape(*orig_shape[:-1], out_c)

    return _KAN()


class KANO1D(BaseOperator):
    """1D FNO 백본 + KAN pointwise 변환."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: int = 8,
        width: int = 16,
        grid_size: int = 5,
        n_layers: int = 2,
        max_epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.grid_size = grid_size
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self._model: Any = None
        self._device: Any = None
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
        in_c, out_c, W, M, G = (
            self.in_channels, self.out_channels, self.width,
            self.modes, self.grid_size,
        )
        n_layers = self.n_layers

        class _KANO(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lift = _kan_layer(in_c, W, G)
                specs = nn.ModuleList()
                kan_post = nn.ModuleList()
                layer_idx = 0
                while layer_idx < n_layers:
                    specs.append(_build_spectral_conv_1d(W, W, M))
                    kan_post.append(_kan_layer(W, W, G))
                    layer_idx += 1
                self.specs = specs
                self.kan_post = kan_post
                self.proj = _kan_layer(W, out_c, G)

            def forward(self, x: Any) -> Any:  # (B, N, C_in)
                x = self.lift(x).permute(0, 2, 1)  # (B, W, N)
                layer_idx = 0
                while layer_idx < len(self.specs):
                    sp_out = self.specs[layer_idx](x).permute(0, 2, 1)  # (B, N, W)
                    x = self.kan_post[layer_idx](sp_out).permute(0, 2, 1)
                    layer_idx += 1
                x = x.permute(0, 2, 1)
                return self.proj(x)

        return _KANO()

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(dataset["inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError(f"(B,N,C) 3D 필요: {X.shape}, {Y.shape}")

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
            epoch_loss = 0.0
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
                epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_loss /= max(len(X), 1)
            self.train_losses_.append(epoch_loss)
            epoch_idx += 1

        self.is_fitted = True
        logger.info(
            "KANO1D 학습 완료: grid=%d loss=%.6g",
            self.grid_size,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        self._check_fitted()
        x = np.asarray(inputs["x"], dtype=np.float32)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None, ...]
        with torch.no_grad():
            y = self._model(torch.tensor(x, device=self._device)).cpu().numpy()
        return y[0] if squeeze else y


__all__ = ["KANO1D"]
