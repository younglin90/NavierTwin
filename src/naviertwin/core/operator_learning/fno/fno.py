"""Fourier Neural Operator (FNO) 1D/2D 구현.

PyTorch 로 직접 구현. SpectralConv 블록 + 포인트-와이즈 가중치를 통해
해상도 독립적(함수 공간) PDE 연산자를 학습한다.

References:
    Li et al., "Fourier Neural Operator on Parametric PDEs", ICLR 2021.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.fno.fno import FNO1D
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 64, 1)).astype(np.float32)
    >>> Y = np.sin(X) + 0.1 * rng.standard_normal((100, 64, 1)).astype(np.float32)
    >>> op = FNO1D(in_channels=1, out_channels=1, modes=8, width=16, n_layers=2, max_epochs=3)
    >>> op.fit({"inputs": X, "outputs": Y})
    >>> y_hat = op.predict({"x": X[:5]})
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _build_spectral_conv_1d(in_c: int, out_c: int, modes: int) -> Any:
    import torch
    import torch.nn as nn

    class SpectralConv1d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scale = 1.0 / (in_c * out_c)
            self.weights = nn.Parameter(
                scale * torch.randn(in_c, out_c, modes, dtype=torch.cfloat)
            )
            self.modes = modes
            self.in_c = in_c
            self.out_c = out_c

        def forward(self, x: Any) -> Any:  # (B, C_in, N)
            B, _, N = x.shape
            x_ft = torch.fft.rfft(x, norm="forward")
            out_ft = torch.zeros(
                B, self.out_c, x_ft.size(-1), dtype=torch.cfloat, device=x.device
            )
            m = min(self.modes, x_ft.size(-1))
            out_ft[:, :, :m] = torch.einsum(
                "bix,iox->box", x_ft[:, :, :m], self.weights[:, :, :m]
            )
            return torch.fft.irfft(out_ft, n=N, norm="forward")

    return SpectralConv1d()


def _build_spectral_conv_2d(in_c: int, out_c: int, modes1: int, modes2: int) -> Any:
    import torch
    import torch.nn as nn

    class SpectralConv2d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scale = 1.0 / (in_c * out_c)
            self.w1 = nn.Parameter(
                scale * torch.randn(in_c, out_c, modes1, modes2, dtype=torch.cfloat)
            )
            self.w2 = nn.Parameter(
                scale * torch.randn(in_c, out_c, modes1, modes2, dtype=torch.cfloat)
            )
            self.modes1 = modes1
            self.modes2 = modes2
            self.out_c = out_c

        def forward(self, x: Any) -> Any:  # (B, C_in, H, W)
            B, _, H, W = x.shape
            x_ft = torch.fft.rfft2(x, norm="forward")
            out_ft = torch.zeros(
                B, self.out_c, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
            )
            m1 = min(self.modes1, H)
            m2 = min(self.modes2, W // 2 + 1)
            out_ft[:, :, :m1, :m2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2]
            )
            out_ft[:, :, -m1:, :m2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2]
            )
            return torch.fft.irfft2(out_ft, s=(H, W), norm="forward")

    return SpectralConv2d()


class FNO1D(BaseOperator):
    """1D FNO — 입력 shape = (B, N, C_in), 출력 shape = (B, N, C_out)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: int = 8,
        width: int = 32,
        n_layers: int = 4,
        max_epochs: int = 100,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
        epoch_callback: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        # epoch(1-index), loss 를 받는 선택적 콜백 (라이브 진행 스트리밍).
        self.epoch_callback = epoch_callback

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

        in_c = self.in_channels
        out_c = self.out_channels
        W = self.width
        M = self.modes

        class _FNO1D(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lift = nn.Linear(in_c, W)
                specs = nn.ModuleList()
                ws = nn.ModuleList()
                layer_idx = 0
                while layer_idx < self.n_layers_:
                    specs.append(_build_spectral_conv_1d(W, W, M))
                    ws.append(nn.Conv1d(W, W, 1))
                    layer_idx += 1
                self.specs = specs
                self.ws = ws
                self.proj1 = nn.Linear(W, 4 * W)
                self.proj2 = nn.Linear(4 * W, out_c)

            def forward(self, x: Any) -> Any:  # (B, N, C_in)
                x = self.lift(x).permute(0, 2, 1)  # (B, W, N)
                layer_idx = 0
                while layer_idx < len(self.specs):
                    x = torch.nn.functional.gelu(
                        self.specs[layer_idx](x) + self.ws[layer_idx](x)
                    )
                    layer_idx += 1
                x = x.permute(0, 2, 1)  # (B, N, W)
                x = torch.nn.functional.gelu(self.proj1(x))
                return self.proj2(x)

        _FNO1D.n_layers_ = self.n_layers
        return _FNO1D()

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(dataset["inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError(
                f"inputs/outputs 는 (B,N,C) 3D 여야 합니다: {X.shape}, {Y.shape}"
            )
        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        xb_t = torch.tensor(X)
        yb_t = torch.tensor(Y)
        loader = DataLoader(
            TensorDataset(xb_t, yb_t),
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
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_loss /= max(len(X), 1)
            self.train_losses_.append(epoch_loss)
            epoch_idx += 1
            if self.epoch_callback is not None:
                self.epoch_callback(epoch_idx, epoch_loss)

        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "FNO1D 학습 완료: layers=%d, width=%d, modes=%d, loss=%.6g",
            self.n_layers,
            self.width,
            self.modes,
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


class FNO2D(BaseOperator):
    """2D FNO — 입력 shape = (B, H, W, C_in), 출력 shape = (B, H, W, C_out)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes1: int = 8,
        modes2: int = 8,
        width: int = 16,
        n_layers: int = 4,
        max_epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
        epoch_callback: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.epoch_callback = epoch_callback

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

        in_c, out_c, W, M1, M2 = (
            self.in_channels,
            self.out_channels,
            self.width,
            self.modes1,
            self.modes2,
        )
        n_layers = self.n_layers

        class _FNO2D(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lift = nn.Linear(in_c, W)
                specs = nn.ModuleList()
                ws = nn.ModuleList()
                layer_idx = 0
                while layer_idx < n_layers:
                    specs.append(_build_spectral_conv_2d(W, W, M1, M2))
                    ws.append(nn.Conv2d(W, W, 1))
                    layer_idx += 1
                self.specs = specs
                self.ws = ws
                self.proj1 = nn.Linear(W, 4 * W)
                self.proj2 = nn.Linear(4 * W, out_c)

            def forward(self, x: Any) -> Any:  # (B, H, W, C_in)
                x = self.lift(x).permute(0, 3, 1, 2)  # (B, W, H, W_)
                layer_idx = 0
                while layer_idx < len(self.specs):
                    x = torch.nn.functional.gelu(
                        self.specs[layer_idx](x) + self.ws[layer_idx](x)
                    )
                    layer_idx += 1
                x = x.permute(0, 2, 3, 1)
                x = torch.nn.functional.gelu(self.proj1(x))
                return self.proj2(x)

        return _FNO2D()

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(dataset["inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if X.ndim != 4 or Y.ndim != 4:
            raise ValueError(
                f"inputs/outputs 는 (B,H,W,C) 4D 여야 합니다: {X.shape}, {Y.shape}"
            )

        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

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
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_loss /= max(len(X), 1)
            self.train_losses_.append(epoch_loss)
            epoch_idx += 1
            if self.epoch_callback is not None:
                self.epoch_callback(epoch_idx, epoch_loss)

        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "FNO2D 학습 완료: loss=%.6g",
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        self._check_fitted()
        x = np.asarray(inputs["x"], dtype=np.float32)
        squeeze = x.ndim == 3
        if squeeze:
            x = x[None, ...]
        with torch.no_grad():
            y = self._model(torch.tensor(x, device=self._device)).cpu().numpy()
        return y[0] if squeeze else y


__all__ = ["FNO1D", "FNO2D"]
