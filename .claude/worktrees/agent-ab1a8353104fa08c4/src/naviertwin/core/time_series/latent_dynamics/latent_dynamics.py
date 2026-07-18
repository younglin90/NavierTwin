"""Latent Space Dynamics — AE 잠재공간에서 Neural ODE 로 시간 적분.

고차원 유동장 x(t) 를 AE 로 z(t) 로 압축하고, 잠재공간 동역학
dz/dt = f_θ(z) 를 학습. 예측 시:
    z_0 = enc(x_0)
    z(t_k) = odeint(f, z_0, ...)
    x(t_k) = dec(z(t_k))

References:
    Gonzalez & Balajewicz 2018; Lee & Carlberg 2020.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.time_series.latent_dynamics.latent_dynamics import (
    ...     LatentDynamicsForecaster,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> seqs = rng.standard_normal((4, 30, 20)).astype(np.float32)
    >>> m = LatentDynamicsForecaster(n_features=20, latent=4, hidden=16, max_epochs=3)
    >>> m.fit({"sequences": seqs, "dt": 0.1})
    >>> m.predict(seqs[0, 0], n_steps=5).shape
    (5, 20)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.time_series.base import BaseTimeSeries
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class LatentDynamicsForecaster(BaseTimeSeries):
    """AE + Neural ODE 결합 잠재공간 예측.

    Attributes:
        latent: 잠재 차원.
        hidden: AE/Field MLP 은닉 폭.
        field_hidden: 잠재 동역학 MLP 은닉 폭.
    """

    def __init__(
        self,
        n_features: int,
        latent: int = 8,
        hidden: int = 32,
        field_hidden: int = 32,
        max_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        dt: float = 0.1,
        w_rec: float = 1.0,
        w_pred: float = 1.0,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.n_features = n_features
        self.latent = latent
        self.hidden = hidden
        self.field_hidden = field_hidden
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dt = dt
        self.w_rec = w_rec
        self.w_pred = w_pred
        self.seed = seed
        self.lookback = 1

        self._enc: Any = None
        self._dec: Any = None
        self._field: Any = None
        self._device: Any = None
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
        n_feat, h, z, fh = (
            self.n_features,
            self.hidden,
            self.latent,
            self.field_hidden,
        )

        self._enc = nn.Sequential(
            nn.Linear(n_feat, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, z),
        )
        self._dec = nn.Sequential(
            nn.Linear(z, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, n_feat),
        )
        self._field = nn.Sequential(
            nn.Linear(z, fh), nn.Tanh(),
            nn.Linear(fh, fh), nn.Tanh(),
            nn.Linear(fh, z),
        )

    def _rk4(self, f: Any, x: Any, dt: float) -> Any:
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seqs = np.asarray(dataset["sequences"], dtype=np.float32)
        if seqs.ndim != 3:
            raise ValueError(f"sequences (N,T,F) 3D 필요: {seqs.shape}")
        if seqs.shape[2] != self.n_features:
            raise ValueError(
                f"sequences F({seqs.shape[2]}) != n_features({self.n_features})"
            )
        dt = float(dataset.get("dt", self.dt))

        Xa = np.ascontiguousarray(seqs[:, :-1, :]).reshape(-1, self.n_features)
        Ya = np.ascontiguousarray(seqs[:, 1:, :]).reshape(-1, self.n_features)

        self._device = self._resolve_device()
        self._build()
        self._enc.to(self._device)
        self._dec.to(self._device)
        self._field.to(self._device)

        params = (
            list(self._enc.parameters())
            + list(self._dec.parameters())
            + list(self._field.parameters())
        )
        optim = torch.optim.Adam(params, lr=self.lr)
        mse = torch.nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(Xa), torch.tensor(Ya)),
            batch_size=min(self.batch_size, len(Xa)),
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
                z_t = self._enc(xb)
                x_rec = self._dec(z_t)
                z_next = self._rk4(self._field, z_t, dt)
                x_pred = self._dec(z_next)
                loss = self.w_rec * mse(x_rec, xb) + self.w_pred * mse(x_pred, yb)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_loss /= max(len(Xa), 1)
            self.train_losses_.append(epoch_loss)
            epoch_idx += 1

        self.dt = dt
        self.is_fitted = True
        logger.info(
            "LatentDynamics 학습 완료: latent=%d dt=%.3g loss=%.6g",
            self.latent,
            dt,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, initial_state: np.ndarray, n_steps: int) -> np.ndarray:
        import torch

        self._check_fitted()
        if n_steps < 1:
            raise ValueError(f"n_steps 는 1 이상: {n_steps}")

        x0 = np.asarray(initial_state, dtype=np.float32).ravel()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            z = self._enc(torch.tensor(x0[None, :], device=self._device))
            step = 0
            while step < n_steps:
                z = self._rk4(self._field, z, self.dt)
                x = self._dec(z).cpu().numpy()[0]
                preds.append(x.copy())
                step += 1
        return np.stack(preds)


__all__ = ["LatentDynamicsForecaster"]
