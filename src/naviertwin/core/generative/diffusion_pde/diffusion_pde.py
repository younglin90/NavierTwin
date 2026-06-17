"""Score-based Diffusion 유동장 생성 (DDPM 형태).

x_0 → x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε   (forward noising)
모델 ε_θ(x_t, t) 학습 → 역과정 샘플링으로 새 유동장 생성.

주어진 유동장 스냅샷 집합에서 분포를 학습하고, sample() 으로 새
스냅샷을 생성할 수 있다. 입력: (n_samples, n_features).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.generative.diffusion_pde.diffusion_pde import (
    ...     DiffusionPDE,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 20)).astype(np.float32)
    >>> m = DiffusionPDE(n_features=20, hidden=32, n_steps=20, max_epochs=3)
    >>> m.fit(X)
    >>> m.sample(n_samples=4, seed=1).shape
    (4, 20)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class DiffusionPDE:
    """DDPM-style 유동장 생성기.

    Attributes:
        n_features: 스냅샷 차원.
        hidden: denoiser MLP 폭.
        n_steps: diffusion 단계 T.
        beta_start / beta_end: 노이즈 스케줄.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        n_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        max_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        if n_steps < 2:
            raise ValueError("n_steps >= 2 필요")
        self.n_features = n_features
        self.hidden = hidden
        self.n_steps = n_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed

        self._model: Any = None
        self._device: Any = None
        self.is_fitted: bool = False
        self.train_losses_: list[float] = []

        # schedule (populated at fit-time on device)
        self._betas: Any = None
        self._alphas: Any = None
        self._alpha_bars: Any = None

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

        n_feat, h = self.n_features, self.hidden

        class _Denoiser(nn.Module):
            """입력 (x, t_norm) → 예측 노이즈 ε."""

            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_feat + 1, h), nn.GELU(),
                    nn.Linear(h, h), nn.GELU(),
                    nn.Linear(h, h), nn.GELU(),
                    nn.Linear(h, n_feat),
                )

            def forward(self, x: Any, t_norm: Any) -> Any:
                return self.net(torch.cat([x, t_norm], dim=-1))

        return _Denoiser()

    def _make_schedule(self) -> None:
        import torch

        betas = torch.linspace(
            self.beta_start, self.beta_end, self.n_steps, device=self._device
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self._betas = betas
        self._alphas = alphas
        self._alpha_bars = alpha_bars

    def fit(self, X: NDArray[np.float64]) -> None:
        """(n_samples, n_features) 스냅샷 집합으로 denoiser 학습."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features:
            raise ValueError(
                f"X shape={X_arr.shape} — (n, {self.n_features}) 2D 필요"
            )

        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        self._make_schedule()

        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        mse = torch.nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(X_arr)),
            batch_size=min(self.batch_size, len(X_arr)),
            shuffle=True,
        )

        self.train_losses_ = []
        epoch_idx = 0
        while epoch_idx < self.max_epochs:
            epoch_loss = 0.0
            batches = iter(loader)
            while True:
                try:
                    (xb,) = next(batches)
                except StopIteration:
                    break
                xb = xb.to(self._device)
                B = xb.shape[0]
                t = torch.randint(0, self.n_steps, (B,), device=self._device)
                alpha_bar = self._alpha_bars[t].unsqueeze(-1)
                eps = torch.randn_like(xb)
                x_t = torch.sqrt(alpha_bar) * xb + torch.sqrt(1.0 - alpha_bar) * eps
                t_norm = (t.float() / max(self.n_steps - 1, 1)).unsqueeze(-1)

                optim.zero_grad()
                eps_hat = self._model(x_t, t_norm)
                loss = mse(eps_hat, eps)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * B
            epoch_loss /= max(len(X_arr), 1)
            self.train_losses_.append(epoch_loss)
            epoch_idx += 1

        self.is_fitted = True
        logger.info(
            "DiffusionPDE 학습 완료: T=%d, loss=%.6g",
            self.n_steps,
            self.train_losses_[-1],
        )

    def sample(
        self, n_samples: int, seed: int | None = None
    ) -> NDArray[np.float64]:
        """역과정 샘플링으로 새 스냅샷을 생성한다."""
        import torch

        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출하세요")

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self._device).manual_seed(seed)
        x = torch.randn(
            (n_samples, self.n_features), device=self._device, generator=gen
        )
        with torch.no_grad():
            t = self.n_steps - 1
            while t >= 0:
                t_tensor = torch.full((n_samples,), t, device=self._device)
                alpha = self._alphas[t]
                alpha_bar = self._alpha_bars[t]
                beta = self._betas[t]

                t_norm = (t_tensor.float() / max(self.n_steps - 1, 1)).unsqueeze(-1)
                eps_hat = self._model(x, t_norm)

                coef = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar + 1e-12)
                mean = (x - coef * eps_hat) / torch.sqrt(alpha + 1e-12)
                if t > 0:
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(beta) * noise
                else:
                    x = mean
                t -= 1
        return x.cpu().numpy().astype(np.float64)


__all__ = ["DiffusionPDE"]
