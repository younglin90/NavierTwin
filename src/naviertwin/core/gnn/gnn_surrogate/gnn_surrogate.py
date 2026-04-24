"""GNN Surrogate — 비정형 메쉬 직접 입력 신경 연산자.

PyTorch Geometric 기반. 노드 특징 → GCN 메시지패싱 → 노드별 필드 예측.
메쉬가 바뀌어도 edge_index 만 갱신하면 재학습 없이 새 메쉬에 적용 가능.

dataset 규약 (fit):
    - ``"node_features"``: (n_samples, n_nodes, f_in) — 각 샘플 노드 특징.
    - ``"outputs"``: (n_samples, n_nodes, f_out) — 목표 필드.
    - ``"edge_index"``: (2, n_edges) 공유 에지 (메쉬 동일 가정).

predict 규약:
    - ``"x"``: (n_nodes, f_in) 또는 (B, n_nodes, f_in).
    - ``"edge_index"``: 선택 — 없으면 fit 시 edge 재사용.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.gnn.gnn_surrogate.gnn_surrogate import GNNSurrogate
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 30, 2)).astype(np.float32)
    >>> Y = X ** 2
    >>> edge = np.stack([np.arange(30), np.roll(np.arange(30), -1)]).astype(np.int64)
    >>> op = GNNSurrogate(in_dim=2, out_dim=2, hidden=16, n_layers=2, max_epochs=3)
    >>> op.fit({"node_features": X, "outputs": Y, "edge_index": edge})
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


def _require_pyg() -> Any:
    try:
        import torch_geometric
    except ImportError as exc:
        raise RuntimeError(_PYG_MISSING) from exc
    return torch_geometric


class GNNSurrogate(BaseOperator):
    """3-layer GCN surrogate."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 32,
        n_layers: int = 3,
        max_epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self._model: Any = None
        self._edge: Any = None
        self._device: Any = None
        self.train_losses_: list[float] = []

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self) -> Any:
        _require_pyg()
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GCNConv

        if self.seed is not None:
            torch.manual_seed(self.seed)

        layers: list[tuple[int, int]] = []
        prev = self.in_dim
        for _ in range(self.n_layers - 1):
            layers.append((prev, self.hidden))
            prev = self.hidden
        layers.append((prev, self.out_dim))

        class _GCN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.convs = nn.ModuleList([GCNConv(a, b) for a, b in layers])

            def forward(self, x: Any, edge_index: Any) -> Any:
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = torch.relu(x)
                return x

        return _GCN()

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch

        X = np.asarray(dataset["node_features"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        edge = np.asarray(dataset["edge_index"], dtype=np.int64)
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError(
                f"node_features/outputs 는 (B,N,F) 3D 필요: {X.shape}, {Y.shape}"
            )
        if edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError(f"edge_index shape 는 (2,E) 이어야 합니다: {edge.shape}")

        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        self._edge = torch.tensor(edge, dtype=torch.long, device=self._device)
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        Xt = torch.tensor(X, device=self._device)
        Yt = torch.tensor(Y, device=self._device)

        self.train_losses_ = []
        for _ in range(self.max_epochs):
            epoch_loss = 0.0
            # 각 샘플을 개별 forward (동일 edge_index 공유)
            # 미니배치가 크지 않으면 이 방식이 단순 & 충분
            order = np.random.permutation(len(X))
            for i in order:
                optim.zero_grad()
                pred = self._model(Xt[i], self._edge)
                loss = loss_fn(pred, Yt[i])
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item())
            epoch_loss /= max(len(X), 1)
            self.train_losses_.append(epoch_loss)

        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "GNNSurrogate 학습 완료: n=%d loss=%.6g",
            len(X),
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        self._check_fitted()
        x = np.asarray(inputs["x"], dtype=np.float32)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None, ...]
        edge = inputs.get("edge_index")
        if edge is None:
            edge_t = self._edge
        else:
            edge_t = torch.tensor(
                np.asarray(edge, dtype=np.int64),
                dtype=torch.long,
                device=self._device,
            )
        preds: list[np.ndarray] = []
        xt = torch.tensor(x, device=self._device)
        with torch.no_grad():
            for i in range(x.shape[0]):
                out = self._model(xt[i], edge_t)
                preds.append(out.cpu().numpy())
        arr = np.stack(preds)
        return arr[0] if squeeze else arr


__all__ = ["GNNSurrogate"]
