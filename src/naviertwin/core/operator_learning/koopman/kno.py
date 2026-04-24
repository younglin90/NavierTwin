"""KNO (Koopman Neural Operator) — 비선형 동역학의 선형 예측.

    x → φ(x) (encoder)
    z_{t+1} = K · z_t      (잠재 선형 진화 — Koopman 연산자 K)
    x_hat = ψ(z) (decoder)

손실: reconstruction(x, ψ(φ(x))) + prediction(x_{t+1}, ψ(K φ(x_t))).

References:
    Xiong et al., "Koopman Neural Operator as a Mesh-Free Solver", JCP 2024.

dataset 규약 (fit):
    - ``"sequences"``: (n_seq, T, F) 시계열 스냅샷.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.koopman.kno import KNO
    >>> rng = np.random.default_rng(0)
    >>> seqs = rng.standard_normal((4, 30, 3)).astype(np.float32)
    >>> m = KNO(n_features=3, latent=6, max_epochs=3)
    >>> m.fit({"sequences": seqs})
    >>> m.predict(seqs[0, 0], n_steps=10).shape
    (10, 3)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.time_series.base import BaseTimeSeries
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class KNO(BaseTimeSeries):
    """Learned Koopman operator with latent encoder/decoder."""

    def __init__(
        self,
        n_features: int,
        latent: int = 16,
        hidden: int = 32,
        max_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_pred: float = 1.0,
        weight_rec: float = 1.0,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.n_features = n_features
        self.latent = latent
        self.hidden = hidden
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_pred = weight_pred
        self.weight_rec = weight_rec
        self.seed = seed
        self.lookback = 1

        self._enc: Any = None
        self._dec: Any = None
        self._K: Any = None
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
        n_feat, h, z = self.n_features, self.hidden, self.latent

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
        # 선형 Koopman 연산자 K (latent × latent)
        self._K = nn.Linear(z, z, bias=False)

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seqs = np.asarray(dataset["sequences"], dtype=np.float32)
        if seqs.ndim != 3:
            raise ValueError(f"sequences (N,T,F) 3D 필요: {seqs.shape}")

        X: list[np.ndarray] = []
        Y: list[np.ndarray] = []
        for s in seqs:
            for t in range(s.shape[0] - 1):
                X.append(s[t])
                Y.append(s[t + 1])
        Xa = np.asarray(X, dtype=np.float32)
        Ya = np.asarray(Y, dtype=np.float32)

        self._device = self._resolve_device()
        self._build()
        self._enc.to(self._device)
        self._dec.to(self._device)
        self._K.to(self._device)

        params = (
            list(self._enc.parameters())
            + list(self._dec.parameters())
            + list(self._K.parameters())
        )
        optim = torch.optim.Adam(params, lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(Xa), torch.tensor(Ya)),
            batch_size=min(self.batch_size, len(Xa)),
            shuffle=True,
        )
        self.train_losses_ = []
        for _ in range(self.max_epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                z = self._enc(xb)
                x_rec = self._dec(z)
                z_next = self._K(z)
                x_pred = self._dec(z_next)
                loss = (
                    self.weight_rec * loss_fn(x_rec, xb)
                    + self.weight_pred * loss_fn(x_pred, yb)
                )
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_loss /= max(len(Xa), 1)
            self.train_losses_.append(epoch_loss)

        self.is_fitted = True
        logger.info(
            "KNO 학습 완료: latent=%d loss=%.6g",
            self.latent,
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
            for _ in range(n_steps):
                z = self._K(z)
                x = self._dec(z).cpu().numpy()[0]
                preds.append(x.copy())
        return np.stack(preds)

    def koopman_matrix(self) -> np.ndarray:
        """학습된 Koopman 행렬 K (latent × latent) 를 numpy 로 반환."""
        self._check_fitted()
        return self._K.weight.detach().cpu().numpy()


__all__ = ["KNO"]
