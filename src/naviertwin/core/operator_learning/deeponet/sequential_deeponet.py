"""Sequential DeepONet — 시간 의존 하중의 순차 예측.

branch(u_t): 과거 H 스텝 입력 함수 → 잠재, trunk(y, t): 쿼리 + 시간.
ROM 시계열 자기회귀 예측 인터페이스.

dataset 규약 (fit):
    - ``"branch_inputs"``: (n_samples, H, m) — 과거 H 스텝 입력.
    - ``"trunk_inputs"``: (q, d) — 쿼리 좌표.
    - ``"outputs"``: (n_samples, q) — 다음 스텝 G(u)(y).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.deeponet.sequential_deeponet import (
    ...     SequentialDeepONet,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> B = rng.standard_normal((30, 5, 8)).astype(np.float32)
    >>> T = rng.standard_normal((12, 1)).astype(np.float32)
    >>> Y = rng.standard_normal((30, 12)).astype(np.float32)
    >>> op = SequentialDeepONet(
    ...     m=8, history=5, trunk_in=1, hidden=16, latent=8, max_epochs=2,
    ... )
    >>> op.fit({"branch_inputs": B, "trunk_inputs": T, "outputs": Y})
    >>> op.predict({"branch_inputs": B[:3], "trunk_inputs": T}).shape
    (3, 12)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.core.operator_learning.deeponet.deeponet import _mlp
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class SequentialDeepONet(BaseOperator):
    """GRU-기반 branch + MLP trunk."""

    def __init__(
        self,
        m: int,
        history: int,
        trunk_in: int,
        hidden: int = 64,
        latent: int = 32,
        n_trunk_layers: int = 3,
        activation: str = "gelu",
        max_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.m = m
        self.history = history
        self.trunk_in = trunk_in
        self.hidden = hidden
        self.latent = latent
        self.n_trunk_layers = n_trunk_layers
        self.activation = activation
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self._branch: Any = None
        self._trunk: Any = None
        self._bias: Any = None
        self._device: Any = None
        self._trunk_cache: Any = None
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

        class _BranchGRU(nn.Module):
            def __init__(self, m: int, h: int, z: int) -> None:
                super().__init__()
                self.rnn = nn.GRU(m, h, batch_first=True)
                self.head = nn.Linear(h, z)

            def forward(self, x: Any) -> Any:  # (B, H, m)
                out, _ = self.rnn(x)
                return self.head(out[:, -1, :])

        self._branch = _BranchGRU(self.m, self.hidden, self.latent)
        self._trunk = _mlp(
            [self.trunk_in] + [self.hidden] * (self.n_trunk_layers - 1) + [self.latent],
            self.activation,
        )
        self._branch.register_parameter("bias_scalar", nn.Parameter(torch.zeros(1)))

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        B = np.asarray(dataset["branch_inputs"], dtype=np.float32)
        T = np.asarray(dataset["trunk_inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if B.ndim != 3 or B.shape[1] != self.history or B.shape[2] != self.m:
            raise ValueError(
                f"branch_inputs shape={B.shape}, 기대 (N, {self.history}, {self.m})"
            )
        if Y.shape[1] != T.shape[0]:
            raise ValueError("outputs 쿼리 길이 불일치")

        self._device = self._resolve_device()
        self._build()
        self._branch.to(self._device)
        self._trunk.to(self._device)
        trunk_t = torch.tensor(T, device=self._device)

        params = (
            list(self._branch.parameters())
            + list(self._trunk.parameters())
        )
        optim = torch.optim.Adam(params, lr=self.lr)
        mse = torch.nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(B), torch.tensor(Y)),
            batch_size=min(self.batch_size, len(B)),
            shuffle=True,
        )
        self.train_losses_ = []
        for _ in range(self.max_epochs):
            epoch = 0.0
            for bb, yb in loader:
                bb = bb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                bfeat = self._branch(bb)
                tfeat = self._trunk(trunk_t)
                pred = bfeat @ tfeat.T + self._branch.bias_scalar
                loss = mse(pred, yb)
                loss.backward()
                optim.step()
                epoch += float(loss.item()) * bb.shape[0]
            epoch /= max(len(B), 1)
            self.train_losses_.append(epoch)

        self._trunk_cache = T
        self.is_fitted = True
        logger.info("SequentialDeepONet 학습 완료: loss=%.6g", self.train_losses_[-1])

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        self._check_fitted()
        B = np.asarray(inputs["branch_inputs"], dtype=np.float32)
        T = np.asarray(
            inputs.get("trunk_inputs", self._trunk_cache), dtype=np.float32
        )
        with torch.no_grad():
            bt = torch.tensor(B, device=self._device)
            tt = torch.tensor(T, device=self._device)
            pred = self._branch(bt) @ self._trunk(tt).T + self._branch.bias_scalar
        return pred.cpu().numpy()


__all__ = ["SequentialDeepONet"]
