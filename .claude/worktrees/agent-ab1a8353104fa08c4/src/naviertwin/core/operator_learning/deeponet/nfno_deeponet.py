"""NFNO-DeepONet — 비균일 그리드용 DeepONet + NUFFT-like branch.

불규칙 격자 입력 함수 u(x_j) 를 branch 에 넣기 위해 각 샘플 좌표 x_j 와
값 u_j 를 함께 입력 (set-invariant aggregator).

- branch: {(x_j, u_j)}_{j=1..m} → 잠재. 여기서는 간단히 max-pool 기반
  DeepSets 인코더.
- trunk: 쿼리 좌표 y → 잠재.
- 내적 + bias → G(u)(y).

dataset 규약 (fit):
    - ``"branch_coords"``: (N, m, d)
    - ``"branch_values"``: (N, m)  또는 (N, m, C)
    - ``"trunk_inputs"``: (q, d)
    - ``"outputs"``: (N, q)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.deeponet.nfno_deeponet import NFNODeepONet
    >>> rng = np.random.default_rng(0)
    >>> Bx = rng.uniform(-1, 1, (20, 30, 2)).astype(np.float32)
    >>> Bu = rng.standard_normal((20, 30)).astype(np.float32)
    >>> T  = rng.uniform(-1, 1, (10, 2)).astype(np.float32)
    >>> Y  = rng.standard_normal((20, 10)).astype(np.float32)
    >>> op = NFNODeepONet(branch_coord_dim=2, branch_value_dim=1, trunk_in=2,
    ...                   hidden=16, latent=8, max_epochs=2)
    >>> op.fit({"branch_coords": Bx, "branch_values": Bu, "trunk_inputs": T, "outputs": Y})
    >>> op.predict({"branch_coords": Bx[:3], "branch_values": Bu[:3], "trunk_inputs": T}).shape
    (3, 10)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.core.operator_learning.deeponet.deeponet import _mlp
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class NFNODeepONet(BaseOperator):
    """비균일 격자 branch — DeepSets + max pool 인코더."""

    def __init__(
        self,
        branch_coord_dim: int,
        branch_value_dim: int,
        trunk_in: int,
        hidden: int = 32,
        latent: int = 16,
        n_layers: int = 3,
        max_epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.branch_coord_dim = branch_coord_dim
        self.branch_value_dim = branch_value_dim
        self.trunk_in = trunk_in
        self.hidden = hidden
        self.latent = latent
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self._point_enc: Any = None
        self._trunk: Any = None
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

        point_in = self.branch_coord_dim + self.branch_value_dim
        self._point_enc = _mlp(
            [point_in] + [self.hidden] * (self.n_layers - 1) + [self.latent],
            "gelu",
        )
        self._trunk = _mlp(
            [self.trunk_in] + [self.hidden] * (self.n_layers - 1) + [self.latent],
            "gelu",
        )
        # bias 를 branch 로 등록
        self._point_enc.register_parameter("bias_scalar", nn.Parameter(torch.zeros(1)))

    def _encode_branch(self, coords: Any, values: Any) -> Any:
        """(B, m, d), (B, m[, C]) → (B, latent) — max pool."""
        import torch

        if values.ndim == 2:
            values = values.unsqueeze(-1)
        feat = torch.cat([coords, values], dim=-1)  # (B, m, d+C)
        emb = self._point_enc(feat)  # (B, m, latent)
        return emb.max(dim=1).values

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        Bx = np.asarray(dataset["branch_coords"], dtype=np.float32)
        Bu = np.asarray(dataset["branch_values"], dtype=np.float32)
        T = np.asarray(dataset["trunk_inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if Bx.shape[0] != Bu.shape[0]:
            raise ValueError("branch_coords/values 샘플 수 불일치")
        if Y.shape[1] != T.shape[0]:
            raise ValueError("outputs 쿼리 길이 불일치")

        self._device = self._resolve_device()
        self._build()
        self._point_enc.to(self._device)
        self._trunk.to(self._device)

        trunk_t = torch.tensor(T, device=self._device)
        params = list(self._point_enc.parameters()) + list(self._trunk.parameters())
        optim = torch.optim.Adam(params, lr=self.lr)
        mse = torch.nn.MSELoss()

        loader = DataLoader(
            TensorDataset(
                torch.tensor(Bx), torch.tensor(Bu), torch.tensor(Y)
            ),
            batch_size=min(self.batch_size, len(Bx)),
            shuffle=True,
        )
        self.train_losses_ = []
        epoch_idx = 0
        while epoch_idx < self.max_epochs:
            epoch = 0.0
            batches = iter(loader)
            while True:
                try:
                    bxb, bub, yb = next(batches)
                except StopIteration:
                    break
                bxb = bxb.to(self._device)
                bub = bub.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                bfeat = self._encode_branch(bxb, bub)
                tfeat = self._trunk(trunk_t)
                pred = bfeat @ tfeat.T + self._point_enc.bias_scalar
                loss = mse(pred, yb)
                loss.backward()
                optim.step()
                epoch += float(loss.item()) * bxb.shape[0]
            epoch /= max(len(Bx), 1)
            self.train_losses_.append(epoch)
            epoch_idx += 1

        self._trunk_cache = T
        self.is_fitted = True
        logger.info("NFNO-DeepONet 학습 완료: loss=%.6g", self.train_losses_[-1])

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        self._check_fitted()
        Bx = np.asarray(inputs["branch_coords"], dtype=np.float32)
        Bu = np.asarray(inputs["branch_values"], dtype=np.float32)
        T = np.asarray(
            inputs.get("trunk_inputs", self._trunk_cache), dtype=np.float32
        )
        with torch.no_grad():
            bxt = torch.tensor(Bx, device=self._device)
            but = torch.tensor(Bu, device=self._device)
            tt = torch.tensor(T, device=self._device)
            pred = self._encode_branch(bxt, but) @ self._trunk(tt).T
            pred = pred + self._point_enc.bias_scalar
        return pred.cpu().numpy()


__all__ = ["NFNODeepONet"]
