"""조건부 VAE (cVAE) — 파라미터 조건부 유동장 생성.

인코더 q(z | x, c), 디코더 p(x | z, c). ELBO = recon + β·KL.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.generative.conditional_gen.conditional_gen import (
    ...     ConditionalVAE,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((80, 20)).astype(np.float32)
    >>> C = rng.standard_normal((80, 3)).astype(np.float32)
    >>> cvae = ConditionalVAE(n_features=20, cond_dim=3, latent=4, max_epochs=3)
    >>> cvae.fit(X, C)
    >>> samples = cvae.sample(C[:5])
    >>> samples.shape
    (5, 20)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class ConditionalVAE:
    """조건부 VAE — fit(X, C) / sample(C)."""

    def __init__(
        self,
        n_features: int,
        cond_dim: int,
        latent: int = 8,
        hidden: int = 64,
        beta: float = 1.0,
        max_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        self.n_features = n_features
        self.cond_dim = cond_dim
        self.latent = latent
        self.hidden = hidden
        self.beta = beta
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed

        self._enc_body: Any = None
        self._mu: Any = None
        self._logvar: Any = None
        self._dec: Any = None
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
        F, C, Z, H = self.n_features, self.cond_dim, self.latent, self.hidden

        self._enc_body = nn.Sequential(
            nn.Linear(F + C, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU(),
        )
        self._mu = nn.Linear(H, Z)
        self._logvar = nn.Linear(H, Z)
        self._dec = nn.Sequential(
            nn.Linear(Z + C, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU(),
            nn.Linear(H, F),
        )

    def fit(
        self, X: NDArray[np.float64], C: NDArray[np.float64]
    ) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        Xa = np.asarray(X, dtype=np.float32)
        Ca = np.asarray(C, dtype=np.float32)
        if Xa.shape[0] != Ca.shape[0]:
            raise ValueError("X, C 샘플 수 불일치")

        self._device = self._resolve_device()
        self._build()
        tuple(map(lambda m: m.to(self._device), (self._enc_body, self._mu, self._logvar, self._dec)))

        params = (
            list(self._enc_body.parameters())
            + list(self._mu.parameters())
            + list(self._logvar.parameters())
            + list(self._dec.parameters())
        )
        optim = torch.optim.Adam(params, lr=self.lr)

        loader = DataLoader(
            TensorDataset(torch.tensor(Xa), torch.tensor(Ca)),
            batch_size=min(self.batch_size, len(Xa)),
            shuffle=True,
        )
        self.train_losses_ = []
        epoch_idx = 0
        while epoch_idx < self.max_epochs:
            epoch = 0.0
            batches = iter(loader)
            while True:
                try:
                    xb, cb = next(batches)
                except StopIteration:
                    break
                xb = xb.to(self._device)
                cb = cb.to(self._device)
                optim.zero_grad()
                h = self._enc_body(torch.cat([xb, cb], dim=-1))
                mu = self._mu(h)
                logvar = self._logvar(h)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                x_hat = self._dec(torch.cat([z, cb], dim=-1))
                recon = torch.nn.functional.mse_loss(x_hat, xb, reduction="sum") / xb.shape[0]
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.shape[0]
                loss = recon + self.beta * kl
                loss.backward()
                optim.step()
                epoch += float(loss.item()) * xb.shape[0]
            epoch /= max(len(Xa), 1)
            self.train_losses_.append(epoch)
            epoch_idx += 1

        self.is_fitted = True
        logger.info("cVAE 학습 완료: loss=%.6g", self.train_losses_[-1])

    def sample(
        self, C: NDArray[np.float64], seed: int | None = None
    ) -> NDArray[np.float64]:
        import torch

        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        Ca = np.asarray(C, dtype=np.float32)
        if Ca.ndim == 1:
            Ca = Ca[None, :]
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self._device).manual_seed(seed)
        with torch.no_grad():
            c = torch.tensor(Ca, device=self._device)
            z = torch.randn(
                (c.shape[0], self.latent), device=self._device, generator=gen
            )
            x = self._dec(torch.cat([z, c], dim=-1))
        return x.cpu().numpy().astype(np.float64)


__all__ = ["ConditionalVAE"]
