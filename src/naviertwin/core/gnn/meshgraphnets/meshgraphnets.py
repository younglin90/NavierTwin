"""MeshGraphNets — 시간발전 GNN 시뮬레이터 (간이 구현).

각 타임스텝에서 delta prediction:
    u_{t+1} = u_t + MeshGraphNet(u_t, edge_features)

Encode-Process-Decode 구조:
    - Encoder: node/edge 특징을 잠재 차원으로 매핑
    - Processor: M 회 메시지패싱 (edge MLP + node MLP)
    - Decoder: 노드 잠재 → delta 필드

References:
    Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks", ICML 2021.

dataset 규약 (fit):
    - ``"trajectories"``: (n_traj, T+1, n_nodes, f) 시간발전 스냅샷
    - ``"edge_index"``: (2, E) 공유 에지
    - ``"edge_features"``: (E, e) 선택 — 좌표 차이 등 (없으면 1 상수)

predict 규약:
    - ``"x"``: (n_nodes, f) 초기 상태
    - ``"n_steps"``: int 예측 스텝 수
    - ``"edge_index"``: 선택

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.gnn.meshgraphnets.meshgraphnets import MeshGraphNets
    >>> rng = np.random.default_rng(0)
    >>> traj = rng.standard_normal((4, 5, 20, 2)).astype(np.float32)
    >>> edge = np.stack([np.arange(20), np.roll(np.arange(20), -1)]).astype(np.int64)
    >>> op = MeshGraphNets(node_feat=2, edge_feat=1, hidden=16, n_msgpass=2, max_epochs=3)
    >>> op.fit({"trajectories": traj, "edge_index": edge})
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

_PYG_MISSING = (
    "torch_geometric 설치 필요: pip install naviertwin[full]"
)


class MeshGraphNets(BaseOperator):
    """Encode-Process-Decode MeshGraphNet 경량 구현."""

    def __init__(
        self,
        node_feat: int,
        edge_feat: int = 1,
        hidden: int = 32,
        n_msgpass: int = 4,
        max_epochs: int = 100,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.node_feat = node_feat
        self.edge_feat = edge_feat
        self.hidden = hidden
        self.n_msgpass = n_msgpass
        self.max_epochs = max_epochs
        self.lr = lr
        self.seed = seed

        self._model: Any = None
        self._edge: Any = None
        self._edge_features: Any = None
        self._device: Any = None
        self.train_losses_: list[float] = []

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self) -> Any:
        try:
            import torch_geometric  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(_PYG_MISSING) from exc
        import torch
        import torch.nn as nn
        from torch_geometric.nn import MessagePassing

        if self.seed is not None:
            torch.manual_seed(self.seed)

        h = self.hidden
        n_msg = self.n_msgpass
        node_in = self.node_feat
        edge_in = self.edge_feat

        def _mlp(a: int, b: int, c: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(a, b), nn.ReLU(), nn.Linear(b, c)
            )

        class _EdgeNodeBlock(MessagePassing):
            def __init__(self) -> None:
                super().__init__(aggr="add")
                self.edge_mlp = _mlp(2 * h + h, h, h)
                self.node_mlp = _mlp(2 * h, h, h)

            def forward(self, x: Any, edge_index: Any, edge_attr: Any) -> tuple[Any, Any]:
                edge_out = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
                node_out = self.propagate(edge_index, x=x, edge_attr=edge_out)
                return x + node_out, edge_attr + edge_out

            def edge_update(self, x_i: Any, x_j: Any, edge_attr: Any) -> Any:
                return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

            def message(self, x_i: Any, edge_attr: Any) -> Any:
                return edge_attr

            def update(self, aggr_out: Any, x: Any) -> Any:
                return self.node_mlp(torch.cat([x, aggr_out], dim=-1))

        class _MGN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.node_enc = _mlp(node_in, h, h)
                self.edge_enc = _mlp(edge_in, h, h)
                self.blocks = nn.ModuleList([_EdgeNodeBlock() for _ in range(n_msg)])
                self.node_dec = _mlp(h, h, node_in)

            def forward(self, x: Any, edge_index: Any, edge_attr: Any) -> Any:
                x = self.node_enc(x)
                edge_attr = self.edge_enc(edge_attr)
                for blk in self.blocks:
                    x, edge_attr = blk(x, edge_index, edge_attr)
                return self.node_dec(x)

        return _MGN()

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch

        traj = np.asarray(dataset["trajectories"], dtype=np.float32)
        edge = np.asarray(dataset["edge_index"], dtype=np.int64)
        if traj.ndim != 4:
            raise ValueError(f"trajectories (n_traj, T+1, N, f) 4D 필요: {traj.shape}")

        n_traj, Tp1, n_nodes, f = traj.shape
        if Tp1 < 2:
            raise ValueError("trajectories 타임스텝이 2 이상이어야 합니다")

        edge_feat = dataset.get("edge_features")
        if edge_feat is None:
            edge_feat_np = np.ones((edge.shape[1], self.edge_feat), dtype=np.float32)
        else:
            edge_feat_np = np.asarray(edge_feat, dtype=np.float32)
            if edge_feat_np.shape[1] != self.edge_feat:
                raise ValueError(
                    f"edge_features 차원({edge_feat_np.shape[1]}) != {self.edge_feat}"
                )

        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        self._edge = torch.tensor(edge, dtype=torch.long, device=self._device)
        self._edge_features = torch.tensor(edge_feat_np, device=self._device)

        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        self.train_losses_ = []
        for _ in range(self.max_epochs):
            epoch_loss = 0.0
            order = np.random.permutation(n_traj)
            for ti in order:
                for t in range(Tp1 - 1):
                    x = torch.tensor(traj[ti, t], device=self._device)
                    y = torch.tensor(traj[ti, t + 1], device=self._device)
                    delta_true = y - x
                    optim.zero_grad()
                    delta_pred = self._model(x, self._edge, self._edge_features)
                    loss = loss_fn(delta_pred, delta_true)
                    loss.backward()
                    optim.step()
                    epoch_loss += float(loss.item())
            epoch_loss /= max(n_traj * (Tp1 - 1), 1)
            self.train_losses_.append(epoch_loss)

        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "MeshGraphNets 학습 완료: trajs=%d, msgpass=%d, loss=%.6g",
            n_traj,
            self.n_msgpass,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        """초기 상태에서 n_steps 만큼 roll-out 예측."""
        import torch

        self._check_fitted()
        x0 = np.asarray(inputs["x"], dtype=np.float32)
        n_steps = int(inputs.get("n_steps", 1))
        if x0.ndim != 2:
            raise ValueError(f"x 는 (N, f) 2D 필요: {x0.shape}")

        edge = inputs.get("edge_index")
        edge_t = (
            self._edge
            if edge is None
            else torch.tensor(
                np.asarray(edge, dtype=np.int64),
                dtype=torch.long,
                device=self._device,
            )
        )

        out = [x0.copy()]
        x = torch.tensor(x0, device=self._device)
        with torch.no_grad():
            for _ in range(n_steps):
                delta = self._model(x, edge_t, self._edge_features)
                x = x + delta
                out.append(x.cpu().numpy())
        return np.stack(out)  # (n_steps+1, N, f)


__all__ = ["MeshGraphNets"]
