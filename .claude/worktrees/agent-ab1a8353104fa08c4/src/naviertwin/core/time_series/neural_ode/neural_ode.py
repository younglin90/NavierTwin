"""Neural ODE 시계열 예측.

    dx/dt = f_θ(t, x)

torchdiffeq 가 설치된 경우 odeint 로 고차 Runge-Kutta 적분.
미설치 시 간단한 명시적 RK4 폴백으로 동작.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.time_series.neural_ode.neural_ode import NeuralODEForecaster
    >>> rng = np.random.default_rng(0)
    >>> seqs = rng.standard_normal((3, 30, 2)).astype(np.float32)
    >>> m = NeuralODEForecaster(n_features=2, hidden=16, max_epochs=3)
    >>> m.fit({"sequences": seqs, "dt": 0.1})
    >>> m.predict(seqs[0, 0], n_steps=5).shape
    (5, 2)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.time_series.base import BaseTimeSeries
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class NeuralODEForecaster(BaseTimeSeries):
    """Small MLP 기반 Neural ODE (dt 기반 step-by-step 학습)."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 32,
        n_layers: int = 2,
        max_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        dt: float = 0.1,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.n_features = n_features
        self.hidden = hidden
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dt = dt
        self.seed = seed
        self.lookback = 1

        self._f: Any = None  # vector field MLP
        self._device: Any = None
        self._use_torchdiffeq = False
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

        h = self.hidden
        n_feat = self.n_features

        class _Field(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers: list[nn.Module] = [nn.Linear(n_feat, h), nn.Tanh()]
                layer_idx = 0
                while layer_idx < self.n_hidden_ - 1:
                    layers.extend([nn.Linear(h, h), nn.Tanh()])
                    layer_idx += 1
                layers.append(nn.Linear(h, n_feat))
                self.net = nn.Sequential(*layers)

            def forward(self, t: Any, x: Any) -> Any:
                return self.net(x)

        _Field.n_hidden_ = self.n_layers
        return _Field()

    def _rk4_step(self, f: Any, x: Any, dt: float) -> Any:
        """명시적 RK4 한 스텝."""
        k1 = f(None, x)
        k2 = f(None, x + 0.5 * dt * k1)
        k3 = f(None, x + 0.5 * dt * k2)
        k4 = f(None, x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seqs = np.asarray(dataset["sequences"], dtype=np.float32)
        if seqs.ndim != 3:
            raise ValueError(f"sequences (N,T,F) 3D 필요: {seqs.shape}")
        dt = float(dataset.get("dt", self.dt))

        X_arr = np.ascontiguousarray(seqs[:, :-1, :]).reshape(-1, self.n_features)
        Y_arr = np.ascontiguousarray(seqs[:, 1:, :]).reshape(-1, self.n_features)

        self._device = self._resolve_device()
        self._f = self._build().to(self._device)
        optim = torch.optim.Adam(self._f.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        try:
            from torchdiffeq import odeint_adjoint as odeint  # noqa: F401

            self._use_torchdiffeq = True
        except ImportError:
            self._use_torchdiffeq = False

        loader = DataLoader(
            TensorDataset(torch.tensor(X_arr), torch.tensor(Y_arr)),
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
                    xb, yb = next(batches)
                except StopIteration:
                    break
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                if self._use_torchdiffeq:
                    from torchdiffeq import odeint

                    t_span = torch.tensor([0.0, dt], device=self._device)
                    pred = odeint(self._f, xb, t_span, method="rk4")[-1]
                else:
                    pred = self._rk4_step(self._f, xb, dt)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_loss /= max(len(X_arr), 1)
            self.train_losses_.append(epoch_loss)
            epoch_idx += 1

        self.dt = dt
        self.is_fitted = True
        logger.info(
            "NeuralODE 학습 완료: torchdiffeq=%s, loss=%.6g",
            self._use_torchdiffeq,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, initial_state: np.ndarray, n_steps: int) -> np.ndarray:
        import torch

        self._check_fitted()
        if n_steps < 1:
            raise ValueError(f"n_steps 는 1 이상: {n_steps}")
        x = np.asarray(initial_state, dtype=np.float32).ravel()
        preds: list[np.ndarray] = []
        state = torch.tensor(x, device=self._device)
        with torch.no_grad():
            step = 0
            while step < n_steps:
                if self._use_torchdiffeq:
                    from torchdiffeq import odeint

                    t_span = torch.tensor([0.0, self.dt], device=self._device)
                    state = odeint(self._f, state, t_span, method="rk4")[-1]
                else:
                    state = self._rk4_step(self._f, state, self.dt)
                preds.append(state.cpu().numpy().copy())
                step += 1
        return np.stack(preds)


__all__ = ["NeuralODEForecaster"]
