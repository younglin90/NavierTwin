"""3D U-Net — volumetric 필드 예측.

Input/Output shape = (B, D, H, W, C).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.unet.unet3d import UNet3D
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((2, 8, 8, 8, 1)).astype(np.float32)
    >>> Y = X ** 2
    >>> net = UNet3D(in_channels=1, out_channels=1, base_ch=4, max_epochs=1)
    >>> net.fit({"inputs": X, "outputs": Y})
    >>> net.predict({"x": X[:1]}).shape
    (1, 8, 8, 8, 1)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _conv3d_block(in_c: int, out_c: int) -> Any:
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv3d(in_c, out_c, 3, padding=1),
        nn.GELU(),
        nn.Conv3d(out_c, out_c, 3, padding=1),
        nn.GELU(),
    )


class UNet3D(BaseOperator):
    """2-level volumetric U-Net."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_ch: int = 8,
        max_epochs: int = 20,
        batch_size: int = 2,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_ch = base_ch
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
        c = self.base_ch
        in_c = self.in_channels
        out_c = self.out_channels

        class _U(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.d1 = _conv3d_block(in_c, c)
                self.p1 = nn.MaxPool3d(2)
                self.d2 = _conv3d_block(c, 2 * c)
                self.bot = _conv3d_block(2 * c, 4 * c)
                self.p2 = nn.MaxPool3d(2)
                self.u2 = nn.ConvTranspose3d(4 * c, 2 * c, 2, stride=2)
                self.m2 = _conv3d_block(4 * c, 2 * c)
                self.u1 = nn.ConvTranspose3d(2 * c, c, 2, stride=2)
                self.m1 = _conv3d_block(2 * c, c)
                self.out = nn.Conv3d(c, out_c, 1)

            def forward(self, x: Any) -> Any:
                # x: (B, D, H, W, C) → (B, C, D, H, W)
                x = x.permute(0, 4, 1, 2, 3)
                d1 = self.d1(x)
                d2 = self.d2(self.p1(d1))
                b = self.bot(self.p2(d2))
                u2 = self.u2(b)
                u2 = self.m2(torch.cat([u2, d2], dim=1))
                u1 = self.u1(u2)
                u1 = self.m1(torch.cat([u1, d1], dim=1))
                y = self.out(u1)
                return y.permute(0, 2, 3, 4, 1)

        return _U()

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(dataset["inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if X.ndim != 5 or Y.ndim != 5:
            raise ValueError(f"(B,D,H,W,C) 5D 필요: {X.shape}, {Y.shape}")
        for k in (1, 2, 3):
            if X.shape[k] % 4:
                raise ValueError(
                    f"D, H, W 는 4의 배수여야 합니다 (2-level pooling): {X.shape[1:4]}"
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
        for _ in range(self.max_epochs):
            epoch = 0.0
            for xb, yb in loader:
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

        self.is_fitted = True
        logger.info("UNet3D 학습 완료: loss=%.6g", self.train_losses_[-1])

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        self._check_fitted()
        x = np.asarray(inputs["x"], dtype=np.float32)
        squeeze = x.ndim == 4
        if squeeze:
            x = x[None, ...]
        with torch.no_grad():
            y = self._model(torch.tensor(x, device=self._device)).cpu().numpy()
        return y[0] if squeeze else y


__all__ = ["UNet3D"]
