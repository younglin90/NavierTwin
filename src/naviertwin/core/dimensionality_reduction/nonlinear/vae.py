"""Variational Autoencoder (VAE) 비선형 차원축소.

인코더가 잠재 분포 (μ, log σ²) 를 출력하고, reparameterization trick
으로 샘플링한 z 를 디코더가 복원한다. ELBO = recon_loss + β · KL.

잠재공간 샘플링으로 **새 유동장 생성** 이 가능하다.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.vae import VAE
    >>> X = np.random.default_rng(0).standard_normal((200, 40))
    >>> vae = VAE(latent_dim=4, hidden_dims=[64, 16], max_epochs=5)
    >>> vae.fit(X)
    >>> z = vae.encode(X)        # (40, 4) — 평균 μ
    >>> X_new = vae.sample(n_samples=3)  # 잠재 표준정규에서 샘플
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.dimensionality_reduction.base import BaseReducer
from naviertwin.core.dimensionality_reduction.nonlinear.autoencoder import _build_mlp
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class _VAEEncoder:
    """Encoder wrapper — μ, logσ² 두 헤드를 반환한다."""

    def __init__(self, body: Any, latent_dim: int) -> None:
        import torch.nn as nn

        self.body = body
        # body 의 마지막 Linear out_features 에서 latent 두 헤드로 분기
        last_dim = None
        def capture_linear(m: Any) -> None:
            nonlocal last_dim
            if isinstance(m, nn.Linear):
                last_dim = m.out_features

        tuple(map(capture_linear, body))
        if last_dim is None:
            raise ValueError("Encoder body 에 Linear 가 없습니다.")
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)

    def parameters(self) -> Any:
        yield from self.body.parameters()
        yield from self.fc_mu.parameters()
        yield from self.fc_logvar.parameters()

    def to(self, device: Any) -> "_VAEEncoder":
        self.body.to(device)
        self.fc_mu.to(device)
        self.fc_logvar.to(device)
        return self

    def __call__(self, x: Any) -> tuple[Any, Any]:
        h = self.body(x)
        return self.fc_mu(h), self.fc_logvar(h)


class VAE(BaseReducer):
    """Variational Autoencoder.

    Attributes:
        latent_dim: 잠재 공간 차원.
        hidden_dims: 인코더 은닉층.
        beta: KL 가중치 (β-VAE).
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dims: list[int] | None = None,
        max_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        beta: float = 1.0,
        device: str = "auto",
        activation: str = "relu",
        center: bool = True,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [128, 32]
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.device = device
        self.activation = activation
        self.center = center
        self.seed = seed

        self._encoder: _VAEEncoder | None = None
        self._decoder: Any = None
        self.mean_: NDArray[np.float64] | None = None
        self.train_losses_: list[float] = []
        self._device: Any = None

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self, n_features: int) -> None:
        import torch

        if self.seed is not None:
            torch.manual_seed(self.seed)
        enc_body = _build_mlp(
            [n_features, *self.hidden_dims], self.activation, final_linear=False
        )
        self._encoder = _VAEEncoder(enc_body, self.latent_dim)
        self._decoder = _build_mlp(
            [self.latent_dim, *reversed(self.hidden_dims), n_features],
            self.activation,
        )

    def fit(self, snapshots: NDArray[np.float64]) -> None:
        """스냅샷으로 VAE 를 학습한다.

        Args:
            snapshots: shape = (n_features, n_snapshots).
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if snapshots.ndim != 2:
            raise ValueError(
                f"snapshots 는 2D 이어야 합니다 (shape={snapshots.shape})"
            )

        n_features, n_snapshots = snapshots.shape
        if self.center:
            self.mean_ = snapshots.mean(axis=1).astype(np.float64)
            data = snapshots - self.mean_[:, None]
        else:
            self.mean_ = np.zeros(n_features, dtype=np.float64)
            data = snapshots

        X = torch.tensor(data.T, dtype=torch.float32)
        device = self._resolve_device()
        self._device = device
        self._build(n_features)
        assert self._encoder is not None
        self._encoder.to(device)
        self._decoder.to(device)

        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        optim = torch.optim.Adam(params, lr=self.lr)

        batch = min(self.batch_size, n_snapshots)
        loader = DataLoader(
            TensorDataset(X), batch_size=batch, shuffle=True, drop_last=False
        )

        self.train_losses_ = []
        epoch = 0
        while epoch < self.max_epochs:
            epoch_loss = 0.0
            batches = iter(loader)
            while True:
                try:
                    (xb,) = next(batches)
                except StopIteration:
                    break
                xb = xb.to(device)
                optim.zero_grad()
                mu, logvar = self._encoder(xb)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                x_hat = self._decoder(z)
                recon = torch.nn.functional.mse_loss(x_hat, xb, reduction="sum") / xb.shape[0]
                kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) / xb.shape[0]
                loss = recon + self.beta * kl
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_loss /= max(n_snapshots, 1)
            self.train_losses_.append(epoch_loss)
            epoch += 1

        self.n_components = self.latent_dim
        self.is_fitted = True
        logger.info(
            "VAE 학습 완료: latent=%d, β=%.2f, final_loss=%.6g",
            self.latent_dim,
            self.beta,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def encode(self, snapshots: NDArray[np.float64]) -> NDArray[np.float64]:
        """스냅샷을 잠재 분포의 평균 μ 로 인코딩한다.

        Args:
            snapshots: shape = (n_features, n_snapshots).

        Returns:
            μ shape = (n_snapshots, latent_dim).
        """
        import torch

        self._check_fitted()
        assert self._encoder is not None and self.mean_ is not None

        if snapshots.ndim == 1:
            snapshots = snapshots[:, None]
        data = snapshots - self.mean_[:, None]
        X = torch.tensor(data.T, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            mu, _ = self._encoder(X)
        return mu.cpu().numpy().astype(np.float64)

    def decode(self, coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        """잠재 계수에서 스냅샷을 복원한다.

        Args:
            coefficients: shape = (n_snapshots, latent_dim).

        Returns:
            복원 스냅샷 shape = (n_features, n_snapshots).
        """
        import torch

        self._check_fitted()
        assert self._decoder is not None and self.mean_ is not None

        if coefficients.ndim == 1:
            coefficients = coefficients[None, :]
        Z = torch.tensor(coefficients, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            x_hat = self._decoder(Z)
        rec = x_hat.cpu().numpy().astype(np.float64).T
        return rec + self.mean_[:, None]

    def sample(self, n_samples: int = 1, seed: int | None = None) -> NDArray[np.float64]:
        """잠재 표준정규에서 샘플링하여 새 스냅샷을 생성한다.

        Args:
            n_samples: 생성할 샘플 수.
            seed: numpy rng seed.

        Returns:
            생성된 스냅샷 shape = (n_features, n_samples).
        """
        self._check_fitted()
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_samples, self.latent_dim))
        return self.decode(z)

    @property
    def energy_ratio(self) -> NDArray[np.float64]:
        self._check_fitted()
        return np.linspace(
            1.0 / max(self.latent_dim, 1), 1.0, self.latent_dim, dtype=np.float64
        )
