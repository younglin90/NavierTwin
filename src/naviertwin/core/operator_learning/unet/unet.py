"""2D U-Net 필드 예측 연산자.

입력/출력 shape = (B, H, W, C). 간단한 encoder-decoder + skip connections.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.unet.unet import UNet2D
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 32, 32, 1)).astype(np.float32)
    >>> Y = X ** 2
    >>> net = UNet2D(in_channels=1, out_channels=1, base_ch=8, max_epochs=2)
    >>> net.fit({"inputs": X, "outputs": Y})
    >>> net.predict({"x": X[:2]}).shape
    (2, 32, 32, 1)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _conv_block(in_c: int, out_c: int) -> Any:
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.GELU(),
    )


class UNet2D(BaseOperator):
    """2-level U-Net."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_ch: int = 16,
        max_epochs: int = 50,
        batch_size: int = 8,
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

        class _UNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.down1 = _conv_block(in_c, c)
                self.pool1 = nn.MaxPool2d(2)
                self.down2 = _conv_block(c, 2 * c)
                self.pool2 = nn.MaxPool2d(2)
                self.bottom = _conv_block(2 * c, 4 * c)
                self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
                self.merge2 = _conv_block(4 * c, 2 * c)
                self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
                self.merge1 = _conv_block(2 * c, c)
                self.out = nn.Conv2d(c, out_c, 1)

            def forward(self, x: Any) -> Any:
                # x: (B, H, W, C) → (B, C, H, W)
                x = x.permute(0, 3, 1, 2)
                d1 = self.down1(x)
                d2 = self.down2(self.pool1(d1))
                b = self.bottom(self.pool2(d2))
                u2 = self.up2(b)
                u2 = self.merge2(torch.cat([u2, d2], dim=1))
                u1 = self.up1(u2)
                u1 = self.merge1(torch.cat([u1, d1], dim=1))
                y = self.out(u1)
                return y.permute(0, 2, 3, 1)

        return _UNet()

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(dataset["inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if X.ndim != 4 or Y.ndim != 4:
            raise ValueError(f"(B,H,W,C) 4D 필요: {X.shape}, {Y.shape}")
        if X.shape[1] % 4 or X.shape[2] % 4:
            raise ValueError(
                f"H, W 가 4의 배수여야 합니다(2-level pooling): H={X.shape[1]}, W={X.shape[2]}"
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
        for _ in range(self.max_epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
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

        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "UNet2D 학습 완료: loss=%.6g",
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


__all__ = ["UNet2D"]
