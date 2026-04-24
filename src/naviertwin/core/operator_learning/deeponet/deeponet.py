"""DeepONet (Deep Operator Network) 구현.

Branch net: 입력 함수의 샘플링 값 u(x_1..x_m) → (p,) 잠재.
Trunk net: 쿼리 좌표 y → (p,) 잠재.
출력 G(u)(y) = <branch(u), trunk(y)> + bias.

References:
    Lu et al., "Learning nonlinear operators via DeepONet", Nature MI 2021.

dataset 규약 (fit):
    - ``"branch_inputs"``: (n_samples, m) — 입력 함수 샘플링 값.
    - ``"trunk_inputs"``: (n_query, d) — 쿼리 좌표 (공유).
    - ``"outputs"``: (n_samples, n_query) — G(u)(y) 실제값.

predict 규약:
    - ``"branch_inputs"``: (n_samples, m)
    - ``"trunk_inputs"``: (n_query, d) — 선택 (없으면 학습 시 좌표 재사용).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.deeponet.deeponet import DeepONet
    >>> rng = np.random.default_rng(0)
    >>> m, q = 32, 20
    >>> branch_X = rng.standard_normal((50, m)).astype(np.float32)
    >>> trunk_X = rng.standard_normal((q, 1)).astype(np.float32)
    >>> out = np.sin(branch_X @ rng.standard_normal((m, q))).astype(np.float32)
    >>> op = DeepONet(branch_in=m, trunk_in=1, hidden=32, latent=16, max_epochs=3)
    >>> op.fit({"branch_inputs": branch_X, "trunk_inputs": trunk_X, "outputs": out})
    >>> y_hat = op.predict({"branch_inputs": branch_X[:5], "trunk_inputs": trunk_X})
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _mlp(sizes: list[int], activation: str = "gelu") -> Any:
    import torch.nn as nn

    acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    act_cls = acts.get(activation.lower(), nn.GELU)
    layers: list[Any] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act_cls())
    return nn.Sequential(*layers)


class DeepONet(BaseOperator):
    """표준 DeepONet (unstacked).

    Args:
        branch_in: 입력 함수 샘플링 수 m.
        trunk_in: 쿼리 좌표 차원 d.
        hidden: branch/trunk MLP 은닉 폭.
        latent: 출력 잠재 차원 p.
        n_branch_layers / n_trunk_layers: 깊이.
        activation: 활성화 함수 이름.
    """

    def __init__(
        self,
        branch_in: int,
        trunk_in: int,
        hidden: int = 64,
        latent: int = 32,
        n_branch_layers: int = 3,
        n_trunk_layers: int = 3,
        activation: str = "gelu",
        max_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.branch_in = branch_in
        self.trunk_in = trunk_in
        self.hidden = hidden
        self.latent = latent
        self.n_branch_layers = n_branch_layers
        self.n_trunk_layers = n_trunk_layers
        self.activation = activation
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self._branch: Any = None
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
        b_sizes = [self.branch_in] + [self.hidden] * (self.n_branch_layers - 1) + [self.latent]
        t_sizes = [self.trunk_in] + [self.hidden] * (self.n_trunk_layers - 1) + [self.latent]
        self._branch = _mlp(b_sizes, self.activation)
        self._trunk = _mlp(t_sizes, self.activation)
        # bias 는 branch 모듈에 등록하여 .to(device) 전파를 받는다
        self._branch.register_parameter("bias", nn.Parameter(torch.zeros(1)))

    @property
    def _bias(self) -> Any:
        return self._branch.bias

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        B = np.asarray(dataset["branch_inputs"], dtype=np.float32)
        T = np.asarray(dataset["trunk_inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if B.ndim != 2 or T.ndim != 2 or Y.ndim != 2:
            raise ValueError(
                "branch (B,m), trunk (q,d), outputs (B,q) 2D 필요: "
                f"{B.shape}, {T.shape}, {Y.shape}"
            )
        if Y.shape[1] != T.shape[0]:
            raise ValueError(
                f"outputs 쿼리 길이({Y.shape[1]}) != trunk 길이({T.shape[0]})"
            )

        self._device = self._resolve_device()
        self._build()
        self._branch.to(self._device)
        self._trunk.to(self._device)
        trunk_t = torch.tensor(T, device=self._device)

        params = list(self._branch.parameters()) + list(self._trunk.parameters())
        optim = torch.optim.Adam(params, lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(B), torch.tensor(Y)),
            batch_size=min(self.batch_size, len(B)),
            shuffle=True,
        )
        trunk_feat = None  # cache
        self.train_losses_ = []
        for _ in range(self.max_epochs):
            epoch_loss = 0.0
            trunk_feat = self._trunk(trunk_t)  # (q, p)
            for bb, yb in loader:
                bb = bb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                branch_feat = self._branch(bb)  # (B, p)
                pred = branch_feat @ trunk_feat.T + self._bias
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
                trunk_feat = self._trunk(trunk_t)
                epoch_loss += float(loss.item()) * bb.shape[0]
            epoch_loss /= max(len(B), 1)
            self.train_losses_.append(epoch_loss)

        self._trunk_cache = T
        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "DeepONet 학습 완료: m=%d d=%d p=%d loss=%.6g",
            self.branch_in,
            self.trunk_in,
            self.latent,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        self._check_fitted()
        B = np.asarray(inputs["branch_inputs"], dtype=np.float32)
        T = np.asarray(
            inputs.get("trunk_inputs", self._trunk_cache), dtype=np.float32
        )
        with torch.no_grad():
            bb = torch.tensor(B, device=self._device)
            tt = torch.tensor(T, device=self._device)
            branch_feat = self._branch(bb)
            trunk_feat = self._trunk(tt)
            out = branch_feat @ trunk_feat.T + self._bias
        return out.cpu().numpy()


__all__ = ["DeepONet"]
