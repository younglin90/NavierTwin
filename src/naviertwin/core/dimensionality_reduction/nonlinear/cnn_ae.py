"""CNN Autoencoder — 2D 이미지/유동 필드 압축.

Input shape: (B, H, W, C). 2-level encoder/decoder.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.cnn_ae import CNNAE
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((10, 16, 16, 1)).astype(np.float32)
    >>> ae = CNNAE(H=16, W=16, channels=1, latent_dim=8, max_epochs=3)
    >>> ae.fit(X)
    >>> ae.decode(ae.encode(X)).shape
    (10, 16, 16, 1)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class CNNAE:
    """2D CNN AE — encoder(Conv2d+pool) + decoder(ConvT+upsample)."""

    def __init__(
        self,
        H: int,
        W: int,
        channels: int = 1,
        latent_dim: int = 32,
        base_ch: int = 16,
        max_epochs: int = 50,
        batch_size: int = 8,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        if H % 4 or W % 4:
            raise ValueError(f"H, W 는 4의 배수 (2-level pool): {H}x{W}")
        self.H = H
        self.W = W
        self.channels = channels
        self.latent_dim = latent_dim
        self.base_ch = base_ch
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed

        self._encoder: Any = None
        self._decoder: Any = None
        self._device: Any = None
        self.is_fitted: bool = False
        self.train_losses_: list[float] = []

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self) -> None:
        import torch
        import torch.nn as nn

        if self.seed is not None:
            torch.manual_seed(self.seed)
        c = self.base_ch
        C = self.channels
        Hc = self.H // 4
        Wc = self.W // 4
        flat = 4 * c * Hc * Wc

        self._encoder = nn.Sequential(
            nn.Conv2d(C, c, 3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 4 * c, 3, padding=1), nn.GELU(),
            nn.Flatten(),
            nn.Linear(flat, self.latent_dim),
        )
        self._decoder = nn.Sequential(
            nn.Linear(self.latent_dim, flat),
            nn.Unflatten(1, (4 * c, Hc, Wc)),
            nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2), nn.GELU(),
            nn.ConvTranspose2d(2 * c, c, 2, stride=2), nn.GELU(),
            nn.Conv2d(c, C, 3, padding=1),
        )

    def fit(self, X: NDArray[np.float64]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 4 or X_arr.shape[1] != self.H or X_arr.shape[2] != self.W:
            raise ValueError(f"X shape={X_arr.shape}, (B, {self.H}, {self.W}, {self.channels}) 필요")

        self._device = self._resolve_device()
        self._build()
        self._encoder.to(self._device)
        self._decoder.to(self._device)

        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        optim = torch.optim.Adam(params, lr=self.lr)

        loader = DataLoader(
            TensorDataset(torch.tensor(X_arr)),
            batch_size=min(self.batch_size, len(X_arr)),
            shuffle=True,
        )
        self.train_losses_ = []
        epoch_idx = 0
        while epoch_idx < self.max_epochs:
            epoch = 0.0
            batches = iter(loader)
            while True:
                try:
                    (xb,) = next(batches)
                except StopIteration:
                    break
                xb = xb.to(self._device)
                xb = xb.permute(0, 3, 1, 2)  # (B, C, H, W)
                optim.zero_grad()
                z = self._encoder(xb)
                x_rec = self._decoder(z)
                loss = torch.nn.functional.mse_loss(x_rec, xb)
                loss.backward()
                optim.step()
                epoch += float(loss.item()) * xb.shape[0]
            epoch /= max(len(X_arr), 1)
            self.train_losses_.append(epoch)
            epoch_idx += 1

        self.is_fitted = True
        logger.info("CNNAE 학습 완료: latent=%d, loss=%.6g", self.latent_dim, self.train_losses_[-1])

    def encode(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        import torch

        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        X_arr = np.asarray(X, dtype=np.float32)
        xt = torch.tensor(X_arr, device=self._device).permute(0, 3, 1, 2)
        with torch.no_grad():
            z = self._encoder(xt)
        return z.cpu().numpy().astype(np.float64)

    def decode(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        import torch

        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        zt = torch.tensor(np.asarray(z, dtype=np.float32), device=self._device)
        with torch.no_grad():
            x = self._decoder(zt)
        return x.cpu().permute(0, 2, 3, 1).numpy().astype(np.float64)


__all__ = ["CNNAE"]
