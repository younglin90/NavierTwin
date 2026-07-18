"""Hybrid ROM — POD-Galerkin + NN 잔차 보정.

    x_rec = U a + NN(a)

여기서 U a 는 선형 POD 복원, NN(a) 는 POD 잔차 (x - U a) 를
학습한 경량 MLP 가 예측하는 비선형 보정항.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD
    >>> from naviertwin.core.physics_correction.hybrid_rom import HybridROM
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((80, 40))  # (n_features, n_snapshots)
    >>> pod = SnapshotPOD(n_modes=5)
    >>> pod.fit(X)
    >>> hybrid = HybridROM(reducer=pod, hidden=32, max_epochs=20)
    >>> hybrid.fit(X)
    >>> X_rec = hybrid.reconstruct(X)
    >>> float(np.linalg.norm(X - X_rec) / np.linalg.norm(X)) < 0.1
    True
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class HybridROM:
    """POD-Galerkin + NN 잔차 보정.

    Args:
        reducer: 이미 fit 된 BaseReducer (예: SnapshotPOD).
        hidden: NN 은닉 폭.
        n_layers: NN 은닉 층 수.
        max_epochs, lr: 학습 파라미터.
    """

    def __init__(
        self,
        reducer: Any,
        hidden: int = 64,
        n_layers: int = 2,
        max_epochs: int = 50,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        if not getattr(reducer, "is_fitted", False):
            raise ValueError("reducer 는 fit 이 완료된 상태여야 합니다")
        self.reducer = reducer
        self.hidden = hidden
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.lr = lr
        self.device = device
        self.seed = seed

        self._model: Any = None
        self._device: Any = None
        self.is_fitted: bool = False
        self.train_losses_: list[float] = []

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self, latent_dim: int, n_features: int) -> Any:
        import torch
        import torch.nn as nn

        if self.seed is not None:
            torch.manual_seed(self.seed)

        layers: list[nn.Module] = [nn.Linear(latent_dim, self.hidden), nn.Tanh()]
        layer_idx = 0
        while layer_idx < self.n_layers - 1:
            layers.extend([nn.Linear(self.hidden, self.hidden), nn.Tanh()])
            layer_idx += 1
        layers.append(nn.Linear(self.hidden, n_features))
        return nn.Sequential(*layers)

    def fit(self, snapshots: NDArray[np.float64]) -> None:
        """POD 계수 a 와 잔차 (x - U a) 쌍을 학습한다.

        snapshots: (n_features, n_snapshots).
        """
        import torch

        a = self.reducer.encode(snapshots)  # (n_snap, latent)
        x_lin = self.reducer.decode(a)  # (n_features, n_snap)
        resid = (snapshots - x_lin).T  # (n_snap, n_features)
        n_feat = snapshots.shape[0]

        self._device = self._resolve_device()
        self._model = self._build(a.shape[1], n_feat).to(self._device)
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        mse = torch.nn.MSELoss()

        A = torch.tensor(a, dtype=torch.float32, device=self._device)
        R = torch.tensor(resid, dtype=torch.float32, device=self._device)

        self.train_losses_ = []
        epoch_idx = 0
        while epoch_idx < self.max_epochs:
            optim.zero_grad()
            pred = self._model(A)
            loss = mse(pred, R)
            loss.backward()
            optim.step()
            self.train_losses_.append(float(loss.item()))
            epoch_idx += 1

        self.is_fitted = True
        logger.info(
            "HybridROM 학습 완료: latent=%d, loss=%.6g",
            a.shape[1],
            self.train_losses_[-1],
        )

    def reconstruct(self, snapshots: NDArray[np.float64]) -> NDArray[np.float64]:
        """선형 복원 + NN 잔차 보정 (n_features, n_snapshots)."""
        import torch

        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출하세요")

        a = self.reducer.encode(snapshots)
        x_lin = self.reducer.decode(a)
        A = torch.tensor(a, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            resid = self._model(A).cpu().numpy()
        return x_lin + resid.T


__all__ = ["HybridROM"]
