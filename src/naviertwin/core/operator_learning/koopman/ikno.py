"""IKNO (Invertible Koopman Neural Operator) — Affine coupling blocks.

가역 인코더 ψ(x) 를 Real-NVP 스타일 coupling block 으로 구성하고,
잠재 공간에서 선형 Koopman K 를 학습. decoder 는 ψ⁻¹.

    z_{t+1} = K · ψ(x_t)
    x_{t+1} = ψ⁻¹(z_{t+1})

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.koopman.ikno import IKNO
    >>> rng = np.random.default_rng(0)
    >>> seqs = rng.standard_normal((4, 20, 4)).astype(np.float32)
    >>> m = IKNO(n_features=4, n_blocks=2, hidden=16, max_epochs=3)
    >>> m.fit({"sequences": seqs})
    >>> m.predict(seqs[0, 0], n_steps=5).shape
    (5, 4)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.time_series.base import BaseTimeSeries
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _build_coupling(dim: int, hidden: int) -> Any:
    import torch
    import torch.nn as nn

    class _Coupling(nn.Module):
        """Real-NVP affine coupling — x_a ↔ x_b 번갈아 업데이트."""

        def __init__(self, swap: bool) -> None:
            super().__init__()
            half = dim // 2
            self.half = half
            self.swap = swap
            self.s_net = nn.Sequential(
                nn.Linear(half, hidden), nn.Tanh(),
                nn.Linear(hidden, dim - half),
            )
            self.t_net = nn.Sequential(
                nn.Linear(half, hidden), nn.Tanh(),
                nn.Linear(hidden, dim - half),
            )

        def forward(self, x: Any) -> Any:
            if self.swap:
                xb, xa = x[..., : dim - self.half], x[..., dim - self.half :]
            else:
                xa, xb = x[..., : self.half], x[..., self.half :]
            s = torch.tanh(self.s_net(xa)) * 0.5  # 안정화
            t = self.t_net(xa)
            yb = xb * torch.exp(s) + t
            if self.swap:
                return torch.cat([yb, xa], dim=-1)
            return torch.cat([xa, yb], dim=-1)

        def inverse(self, y: Any) -> Any:
            if self.swap:
                yb, ya = y[..., : dim - self.half], y[..., dim - self.half :]
                xa = ya
            else:
                ya, yb = y[..., : self.half], y[..., self.half :]
                xa = ya
            s = torch.tanh(self.s_net(xa)) * 0.5
            t = self.t_net(xa)
            xb = (yb - t) * torch.exp(-s)
            if self.swap:
                return torch.cat([xb, xa], dim=-1)
            return torch.cat([xa, xb], dim=-1)

    return _Coupling


class IKNO(BaseTimeSeries):
    """Invertible KNO — Real-NVP 인코더 + 선형 Koopman."""

    def __init__(
        self,
        n_features: int,
        n_blocks: int = 4,
        hidden: int = 32,
        max_epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        if n_features < 2:
            raise ValueError("n_features >= 2 필요 (coupling 을 위해)")
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.hidden = hidden
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.lookback = 1

        self._blocks: Any = None
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
        Cp = _build_coupling(self.n_features, self.hidden)
        blocks = nn.ModuleList()
        block_idx = 0
        while block_idx < self.n_blocks:
            blocks.append(Cp(swap=(block_idx % 2 == 1)))
            block_idx += 1
        self._blocks = blocks
        self._K = nn.Linear(self.n_features, self.n_features, bias=False)

    def _encode(self, x: Any) -> Any:
        z = x
        block_idx = 0
        while block_idx < len(self._blocks):
            z = self._blocks[block_idx](z)
            block_idx += 1
        return z

    def _decode(self, z: Any) -> Any:
        x = z
        block_idx = len(self._blocks) - 1
        while block_idx >= 0:
            x = self._blocks[block_idx].inverse(x)
            block_idx -= 1
        return x

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seqs = np.asarray(dataset["sequences"], dtype=np.float32)
        if seqs.ndim != 3:
            raise ValueError(f"sequences (N,T,F) 3D 필요: {seqs.shape}")
        if seqs.shape[2] != self.n_features:
            raise ValueError("F 불일치")

        Xa = np.ascontiguousarray(seqs[:, :-1, :]).reshape(-1, self.n_features)
        Ya = np.ascontiguousarray(seqs[:, 1:, :]).reshape(-1, self.n_features)

        self._device = self._resolve_device()
        self._build()
        self._blocks.to(self._device)
        self._K.to(self._device)

        params = list(self._blocks.parameters()) + list(self._K.parameters())
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
            epoch = 0.0
            batches = iter(loader)
            while True:
                try:
                    xb, yb = next(batches)
                except StopIteration:
                    break
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                z = self._encode(xb)
                z_next = self._K(z)
                x_pred = self._decode(z_next)
                loss = mse(x_pred, yb)
                loss.backward()
                optim.step()
                epoch += float(loss.item()) * xb.shape[0]
            epoch /= max(len(Xa), 1)
            self.train_losses_.append(epoch)
            epoch_idx += 1

        self.is_fitted = True
        logger.info("IKNO 학습 완료: loss=%.6g", self.train_losses_[-1])

    def predict(self, initial_state: np.ndarray, n_steps: int) -> np.ndarray:
        import torch

        self._check_fitted()
        if n_steps < 1:
            raise ValueError("n_steps 는 1 이상")
        x0 = np.asarray(initial_state, dtype=np.float32).ravel()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            x = torch.tensor(x0[None, :], device=self._device)
            z = self._encode(x)
            step = 0
            while step < n_steps:
                z = self._K(z)
                x_next = self._decode(z).cpu().numpy()[0]
                preds.append(x_next.copy())
                step += 1
        return np.stack(preds)


__all__ = ["IKNO"]
