"""HAMLET — Graph Transformer Neural Operator (간이 구현).

노드 위치 임베딩 + multi-head self-attention + residual FFN + node-wise 디코더.
torch_geometric 없이 dense attention 으로 작성 (노드 수가 매우 크면 비효율적이지만
CFD 스냅샷 (~ 10⁴ 이하) 에서는 충분).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.gnn.graph_transformer.hamlet import HAMLET
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((5, 40, 3)).astype(np.float32)
    >>> Y = np.tanh(X).astype(np.float32)
    >>> pos = rng.standard_normal((40, 3)).astype(np.float32)
    >>> op = HAMLET(in_dim=3, out_dim=3, d_model=16, n_heads=2, n_layers=1,
    ...             pos_dim=3, max_epochs=2)
    >>> op.fit({"node_features": X, "outputs": Y, "positions": pos})
    >>> op.predict({"x": X[:2]}).shape
    (2, 40, 3)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class HAMLET(BaseOperator):
    """Graph Transformer Neural Operator wrapper (dense attention)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        pos_dim: int = 3,
        max_epochs: int = 50,
        batch_size: int = 8,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pos_dim = pos_dim
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self._model: Any = None
        self._device: Any = None
        self._positions: Any = None
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

        d = self.d_model
        nh = self.n_heads
        nL = self.n_layers

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
                self.ln1 = nn.LayerNorm(d)
                self.ff = nn.Sequential(
                    nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
                )
                self.ln2 = nn.LayerNorm(d)

            def forward(self, x: Any) -> Any:
                a, _ = self.attn(x, x, x, need_weights=False)
                x = self.ln1(x + a)
                x = self.ln2(x + self.ff(x))
                return x

        class _HAMLET(nn.Module):
            def __init__(self, in_d: int, out_d: int, pos_d: int) -> None:
                super().__init__()
                self.in_proj = nn.Linear(in_d, d)
                self.pos_proj = nn.Linear(pos_d, d)
                blocks = nn.ModuleList()
                block_idx = 0
                while block_idx < nL:
                    blocks.append(_Block())
                    block_idx += 1
                self.blocks = blocks
                self.out_proj = nn.Linear(d, out_d)

            def forward(self, x: Any, pos: Any) -> Any:
                # x: (B, N, in_d), pos: (N, pos_d)
                emb = self.in_proj(x) + self.pos_proj(pos)[None, :, :]
                block_idx = 0
                while block_idx < len(self.blocks):
                    emb = self.blocks[block_idx](emb)
                    block_idx += 1
                return self.out_proj(emb)

        return _HAMLET(self.in_dim, self.out_dim, self.pos_dim)

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(dataset["node_features"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        pos = np.asarray(dataset["positions"], dtype=np.float32)
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError(f"(B,N,d) 3D 필요: {X.shape}, {Y.shape}")
        if X.shape[1] != pos.shape[0]:
            raise ValueError(
                f"N (node count) 불일치: X={X.shape[1]}, pos={pos.shape[0]}"
            )

        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        self._positions = torch.tensor(pos, device=self._device)

        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        mse = torch.nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(Y)),
            batch_size=min(self.batch_size, len(X)),
            shuffle=True,
        )
        self.train_losses_ = []
        epoch_idx = 0
        while epoch_idx < self.max_epochs:
            epoch = 0.0
            batch_iter = iter(loader)
            while True:
                try:
                    xb, yb = next(batch_iter)
                except StopIteration:
                    break
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                pred = self._model(xb, self._positions)
                loss = mse(pred, yb)
                loss.backward()
                optim.step()
                epoch += float(loss.item()) * xb.shape[0]
            epoch /= max(len(X), 1)
            self.train_losses_.append(epoch)
            epoch_idx += 1

        self.is_fitted = True
        logger.info("HAMLET 학습 완료: loss=%.6g", self.train_losses_[-1])

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        self._check_fitted()
        x = np.asarray(inputs["x"], dtype=np.float32)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None, ...]
        with torch.no_grad():
            y = self._model(
                torch.tensor(x, device=self._device), self._positions
            ).cpu().numpy()
        return y[0] if squeeze else y


__all__ = ["HAMLET"]
