"""Fully-connected Autoencoder (AE) 기반 비선형 차원축소.

PyTorch 로 구현한다. CPU/GPU 모두에서 동작하며 입력 스냅샷 규약은
``SnapshotPOD`` 와 동일한 (n_features, n_snapshots).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.autoencoder import (
    ...     Autoencoder,
    ... )
    >>> X = np.random.default_rng(0).standard_normal((200, 30))
    >>> ae = Autoencoder(latent_dim=4, hidden_dims=[64, 16], max_epochs=5)
    >>> ae.fit(X)
    >>> z = ae.encode(X)
    >>> X_rec = ae.decode(z)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.dimensionality_reduction.base import BaseReducer
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _build_mlp(
    sizes: list[int], activation: str = "relu", final_linear: bool = True
) -> Any:
    """주어진 layer 크기로 MLP 시퀀스를 만든다."""
    import torch.nn as nn

    acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    act_cls = acts.get(activation.lower(), nn.ReLU)

    layers: list[Any] = []
    i = 0
    while i < len(sizes) - 1:
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if not (is_last and final_linear):
            layers.append(act_cls())
        i += 1
    return nn.Sequential(*layers)


class Autoencoder(BaseReducer):
    """Fully-connected autoencoder.

    Attributes:
        latent_dim: 잠재 공간 차원.
        hidden_dims: 인코더 은닉층 크기 리스트. 디코더는 대칭 구조.
        max_epochs: 학습 최대 에폭.
        batch_size: 배치 크기.
        lr: 학습률.
        device: "cpu" / "cuda" / "auto".
        activation: "relu" / "gelu" / "tanh".
        center: True 면 평균 필드를 제거한 뒤 학습한다.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dims: list[int] | None = None,
        max_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
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
        self.device = device
        self.activation = activation
        self.center = center
        self.seed = seed

        self._encoder: Any = None
        self._decoder: Any = None
        self.mean_: NDArray[np.float64] | None = None
        self.train_losses_: list[float] = []
        self._n_features: int = 0

    # ------------------------------------------------------------------

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self, n_features: int) -> None:
        import torch

        if self.seed is not None:
            torch.manual_seed(self.seed)
        enc_sizes = [n_features, *self.hidden_dims, self.latent_dim]
        dec_sizes = [self.latent_dim, *reversed(self.hidden_dims), n_features]
        self._encoder = _build_mlp(enc_sizes, self.activation)
        self._decoder = _build_mlp(dec_sizes, self.activation)

    # ------------------------------------------------------------------
    # BaseReducer API
    # ------------------------------------------------------------------

    def fit(self, snapshots: NDArray[np.float64]) -> None:
        """스냅샷으로 AE 를 학습한다.

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
        self._n_features = n_features

        if self.center:
            self.mean_ = snapshots.mean(axis=1).astype(np.float64)
            data = snapshots - self.mean_[:, None]
        else:
            self.mean_ = np.zeros(n_features, dtype=np.float64)
            data = snapshots
        # (n_snapshots, n_features) 로 변환 — PyTorch batch 규약
        X = torch.tensor(data.T, dtype=torch.float32)

        device = self._resolve_device()
        self._build(n_features)
        self._encoder.to(device)
        self._decoder.to(device)

        optim = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=self.lr,
        )
        loss_fn = torch.nn.MSELoss()

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
                z = self._encoder(xb)
                x_hat = self._decoder(z)
                loss = loss_fn(x_hat, xb)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_loss /= max(n_snapshots, 1)
            self.train_losses_.append(epoch_loss)
            if (epoch + 1) % max(1, self.max_epochs // 5) == 0:
                logger.debug("AE epoch %d/%d: loss=%.6g", epoch + 1, self.max_epochs, epoch_loss)
            epoch += 1

        self.n_components = self.latent_dim
        self.is_fitted = True
        logger.info(
            "Autoencoder 학습 완료: latent=%d, final_loss=%.6g",
            self.latent_dim,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def encode(self, snapshots: NDArray[np.float64]) -> NDArray[np.float64]:
        """스냅샷을 잠재 공간으로 인코딩.

        Args:
            snapshots: shape = (n_features, n_snapshots).

        Returns:
            잠재 계수 shape = (n_snapshots, latent_dim).
        """
        import torch

        self._check_fitted()
        assert self.mean_ is not None

        if snapshots.ndim == 1:
            snapshots = snapshots[:, None]
        data = snapshots - self.mean_[:, None]
        X = torch.tensor(data.T, dtype=torch.float32).to(next(self._encoder.parameters()).device)
        with torch.no_grad():
            z = self._encoder(X)
        return z.cpu().numpy().astype(np.float64)

    def decode(self, coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        """잠재 계수를 원래 차원으로 복원.

        Args:
            coefficients: shape = (n_snapshots, latent_dim).

        Returns:
            복원 스냅샷 shape = (n_features, n_snapshots).
        """
        import torch

        self._check_fitted()
        assert self.mean_ is not None

        if coefficients.ndim == 1:
            coefficients = coefficients[None, :]
        Z = torch.tensor(coefficients, dtype=torch.float32).to(next(self._decoder.parameters()).device)
        with torch.no_grad():
            x_hat = self._decoder(Z)
        rec = x_hat.cpu().numpy().astype(np.float64).T
        return rec + self.mean_[:, None]

    @property
    def energy_ratio(self) -> NDArray[np.float64]:
        """AE 는 명시적 에너지 분해가 없으므로 균등 분배를 반환한다."""
        self._check_fitted()
        return np.linspace(
            1.0 / max(self.latent_dim, 1), 1.0, self.latent_dim, dtype=np.float64
        )
