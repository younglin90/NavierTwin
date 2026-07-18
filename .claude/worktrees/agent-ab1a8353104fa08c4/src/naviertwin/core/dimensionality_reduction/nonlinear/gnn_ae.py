"""GNN 기반 Autoencoder (비정형 메쉬용).

PyTorch Geometric 필요 (optional, ``naviertwin[full]``). 미설치 시
import 는 가능하되 ``fit`` 호출 시 RuntimeError 를 발생시킨다.

비정형 메쉬의 각 노드 필드값을 그래프 메시지패싱으로 잠재 공간에
투영·복원한다. 에지 연결성은 사용자가 제공하거나 k-NN 으로 자동 생성.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.gnn_ae import GNNAE
    >>> X = np.random.default_rng(0).standard_normal((120, 40))  # (n_nodes, n_snapshots)
    >>> points = np.random.default_rng(1).standard_normal((120, 3))
    >>> ae = GNNAE(latent_dim=4, hidden_dim=16, max_epochs=5, points=points)
    >>> ae.fit(X)
    >>> z = ae.encode(X)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.dimensionality_reduction.base import BaseReducer
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

_PYG_MISSING = (
    "torch_geometric 설치 필요: pip install naviertwin[full]\n"
    "또는 직접: pip install torch_geometric"
)


def _require_pyg() -> Any:
    try:
        import torch_geometric
    except ImportError as exc:
        raise RuntimeError(_PYG_MISSING) from exc
    return torch_geometric


def _knn_edges(points: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    """k-NN 으로 양방향 에지 인덱스를 생성한다. shape=(2, n_edges)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)
    src = np.repeat(np.arange(len(points)), k)
    dst = idx[:, 1 : k + 1].ravel()
    edges = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return edges.astype(np.int64)


class GNNAE(BaseReducer):
    """간단한 2-layer GraphConv 기반 AE.

    Attributes:
        latent_dim: 잠재 그래프 차원 (노드별 잠재 벡터 크기).
        hidden_dim: 메시지패싱 은닉 차원.
        k: k-NN 에지 개수 (edge_index 가 없을 때).
        points: 노드 좌표 (optional, 에지 계산 시 필요).
        edge_index: 외부 제공 에지. points 와 상충.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 32,
        k: int = 6,
        max_epochs: int = 50,
        lr: float = 1e-3,
        device: str = "auto",
        points: NDArray[np.float64] | None = None,
        edge_index: NDArray[np.int64] | None = None,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.max_epochs = max_epochs
        self.lr = lr
        self.device = device
        self.points = points
        self.edge_index_np = edge_index
        self.seed = seed

        self._encoder: Any = None
        self._decoder: Any = None
        self._edge_tensor: Any = None
        self._device: Any = None
        self.train_losses_: list[float] = []

    def _build(self, n_snap: int) -> None:
        _require_pyg()
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GCNConv

        if self.seed is not None:
            torch.manual_seed(self.seed)

        class _Encoder(nn.Module):
            def __init__(self, in_dim: int, hid: int, out_dim: int) -> None:
                super().__init__()
                self.conv1 = GCNConv(in_dim, hid)
                self.conv2 = GCNConv(hid, out_dim)

            def forward(self, x: Any, edge_index: Any) -> Any:
                h = torch.relu(self.conv1(x, edge_index))
                return self.conv2(h, edge_index)

        class _Decoder(nn.Module):
            def __init__(self, in_dim: int, hid: int, out_dim: int) -> None:
                super().__init__()
                self.conv1 = GCNConv(in_dim, hid)
                self.conv2 = GCNConv(hid, out_dim)

            def forward(self, z: Any, edge_index: Any) -> Any:
                h = torch.relu(self.conv1(z, edge_index))
                return self.conv2(h, edge_index)

        self._encoder = _Encoder(n_snap, self.hidden_dim, self.latent_dim)
        self._decoder = _Decoder(self.latent_dim, self.hidden_dim, n_snap)

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _get_edges(self, n_nodes: int) -> NDArray[np.int64]:
        if self.edge_index_np is not None:
            return np.asarray(self.edge_index_np, dtype=np.int64)
        if self.points is None:
            # 순환 연결 (경로 그래프)
            idx = np.arange(n_nodes)
            return np.stack([idx, np.roll(idx, -1)]).astype(np.int64)
        return _knn_edges(self.points, self.k)

    def fit(self, snapshots: NDArray[np.float64]) -> None:
        """(n_nodes, n_snapshots) 스냅샷으로 GNN-AE 학습.

        각 노드가 n_snapshots 차원 feature 를 가진다고 본다.
        """
        import torch

        if snapshots.ndim != 2:
            raise ValueError(f"snapshots shape={snapshots.shape} (2D 필요)")
        n_nodes, n_snap = snapshots.shape
        if n_snap == 0:
            raise ValueError("스냅샷이 0 개입니다")

        device = self._resolve_device()
        self._device = device
        self._build(n_snap)
        self._encoder.to(device)
        self._decoder.to(device)

        edge_np = self._get_edges(n_nodes)
        self._edge_tensor = torch.tensor(edge_np, dtype=torch.long, device=device)
        x = torch.tensor(snapshots, dtype=torch.float32, device=device)

        optim = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            lr=self.lr,
        )
        loss_fn = torch.nn.MSELoss()

        self.train_losses_ = []
        epoch = 0
        while epoch < self.max_epochs:
            optim.zero_grad()
            z = self._encoder(x, self._edge_tensor)
            x_hat = self._decoder(z, self._edge_tensor)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optim.step()
            self.train_losses_.append(float(loss.item()))
            epoch += 1

        self.n_components = self.latent_dim
        self.is_fitted = True
        logger.info(
            "GNN-AE 학습 완료: nodes=%d, snap=%d, latent=%d, loss=%.6g",
            n_nodes,
            n_snap,
            self.latent_dim,
            self.train_losses_[-1],
        )

    def encode(self, snapshots: NDArray[np.float64]) -> NDArray[np.float64]:
        """shape=(n_nodes, n_snap) → 잠재 shape=(n_nodes, latent_dim)."""
        import torch

        self._check_fitted()
        x = torch.tensor(snapshots, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            z = self._encoder(x, self._edge_tensor)
        return z.cpu().numpy().astype(np.float64)

    def decode(self, coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        """shape=(n_nodes, latent_dim) → shape=(n_nodes, n_snap)."""
        import torch

        self._check_fitted()
        z = torch.tensor(coefficients, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            x_hat = self._decoder(z, self._edge_tensor)
        return x_hat.cpu().numpy().astype(np.float64)

    @property
    def energy_ratio(self) -> NDArray[np.float64]:
        self._check_fitted()
        return np.linspace(
            1.0 / max(self.latent_dim, 1), 1.0, self.latent_dim, dtype=np.float64
        )
