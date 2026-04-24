"""MIONet (Multiple-Input Operator Network).

복수 입력 함수를 동시에 받는 DeepONet 일반화:
    G(u_1, u_2, ..., u_K)(y) = <f(branch_1(u_1), ..., branch_K(u_K)), trunk(y)>

여기서 f 는 element-wise 곱 또는 MLP merge.

dataset 규약 (fit):
    - ``"branch_inputs"``: list[(n_samples, m_k)] — K 개의 branch 입력.
    - ``"trunk_inputs"``: (n_query, d)
    - ``"outputs"``: (n_samples, n_query)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.deeponet.mionet import MIONet
    >>> rng = np.random.default_rng(0)
    >>> B1 = rng.standard_normal((40, 8)).astype(np.float32)
    >>> B2 = rng.standard_normal((40, 12)).astype(np.float32)
    >>> T = rng.standard_normal((10, 2)).astype(np.float32)
    >>> Y = rng.standard_normal((40, 10)).astype(np.float32)
    >>> op = MIONet(branch_in_list=[8, 12], trunk_in=2, hidden=16, latent=8, max_epochs=2)
    >>> op.fit({"branch_inputs": [B1, B2], "trunk_inputs": T, "outputs": Y})
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.core.operator_learning.deeponet.deeponet import _mlp
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class MIONet(BaseOperator):
    """Multiple-input operator network (element-wise merge).

    Attributes:
        branch_in_list: 각 branch 입력 차원 리스트 [m_1, m_2, ...].
        merge: "product" (element-wise 곱) 또는 "concat" (MLP 합성).
    """

    def __init__(
        self,
        branch_in_list: list[int],
        trunk_in: int,
        hidden: int = 64,
        latent: int = 32,
        n_layers: int = 3,
        merge: str = "product",
        activation: str = "gelu",
        max_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        if len(branch_in_list) < 2:
            raise ValueError("MIONet 은 2 개 이상의 branch 가 필요합니다")
        self.branch_in_list = branch_in_list
        self.trunk_in = trunk_in
        self.hidden = hidden
        self.latent = latent
        self.n_layers = n_layers
        self.merge = merge
        self.activation = activation
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self._branches: Any = None
        self._trunk: Any = None
        self._merge_net: Any = None
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

        self._branches = nn.ModuleList(
            [
                _mlp(
                    [m] + [self.hidden] * (self.n_layers - 1) + [self.latent],
                    self.activation,
                )
                for m in self.branch_in_list
            ]
        )
        self._trunk = _mlp(
            [self.trunk_in]
            + [self.hidden] * (self.n_layers - 1)
            + [self.latent],
            self.activation,
        )
        if self.merge == "concat":
            in_dim = self.latent * len(self.branch_in_list)
            self._merge_net = _mlp(
                [in_dim, self.hidden, self.latent], self.activation
            )
        else:
            self._merge_net = nn.Identity()
        self._branches.register_parameter("bias", nn.Parameter(torch.zeros(1)))

    @property
    def _bias(self) -> Any:
        return self._branches.bias

    def _combine(self, branch_feats: list[Any]) -> Any:
        import torch

        if self.merge == "product":
            out = branch_feats[0]
            for f in branch_feats[1:]:
                out = out * f
            return out
        # concat
        return self._merge_net(torch.cat(branch_feats, dim=-1))

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        Bs = [np.asarray(b, dtype=np.float32) for b in dataset["branch_inputs"]]
        T = np.asarray(dataset["trunk_inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)

        if len(Bs) != len(self.branch_in_list):
            raise ValueError(
                f"branch_inputs 수({len(Bs)}) != branch_in_list({len(self.branch_in_list)})"
            )
        for k, (b, m) in enumerate(zip(Bs, self.branch_in_list)):
            if b.shape[1] != m:
                raise ValueError(
                    f"branch[{k}] shape[1]={b.shape[1]} != 기대값 {m}"
                )
        if Y.shape[1] != T.shape[0]:
            raise ValueError("outputs 쿼리 길이 불일치")

        self._device = self._resolve_device()
        self._build()
        self._branches.to(self._device)
        self._trunk.to(self._device)
        self._merge_net.to(self._device)
        trunk_t = torch.tensor(T, device=self._device)

        params = (
            list(self._branches.parameters())
            + list(self._trunk.parameters())
            + list(self._merge_net.parameters())
        )
        optim = torch.optim.Adam(params, lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        tensors = [torch.tensor(b) for b in Bs] + [torch.tensor(Y)]
        loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=min(self.batch_size, len(Bs[0])),
            shuffle=True,
        )

        self.train_losses_ = []
        n_branches = len(self.branch_in_list)
        for _ in range(self.max_epochs):
            epoch_loss = 0.0
            trunk_feat = self._trunk(trunk_t)
            for batch in loader:
                bs = [b.to(self._device) for b in batch[:n_branches]]
                yb = batch[-1].to(self._device)
                optim.zero_grad()
                feats = [self._branches[k](bs[k]) for k in range(n_branches)]
                merged = self._combine(feats)
                trunk_feat = self._trunk(trunk_t)
                pred = merged @ trunk_feat.T + self._bias
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * bs[0].shape[0]
            epoch_loss /= max(len(Bs[0]), 1)
            self.train_losses_.append(epoch_loss)

        self._trunk_cache = T
        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "MIONet 학습 완료: K=%d merge=%s loss=%.6g",
            n_branches,
            self.merge,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        self._check_fitted()
        Bs = [np.asarray(b, dtype=np.float32) for b in inputs["branch_inputs"]]
        T = np.asarray(
            inputs.get("trunk_inputs", self._trunk_cache), dtype=np.float32
        )
        with torch.no_grad():
            bs = [torch.tensor(b, device=self._device) for b in Bs]
            tt = torch.tensor(T, device=self._device)
            feats = [self._branches[k](bs[k]) for k in range(len(bs))]
            merged = self._combine(feats)
            tf = self._trunk(tt)
            out = merged @ tf.T + self._bias
        return out.cpu().numpy()


__all__ = ["MIONet"]
