"""Physics-embedded E(n)-Equivariant GNN (경량 구현).

EGNN (Satorras et al., 2021) 스타일 : 노드 좌표와 스칼라 특징을 함께 갱신.

    m_ij = φ_e(h_i, h_j, ||x_i - x_j||²)
    x_i  = x_i + C Σ_{j≠i} (x_i - x_j) · φ_x(m_ij)
    h_i  = φ_h(h_i, Σ_j m_ij)

E(n) 변환 (회전/평행이동) 에 대해 좌표는 동등, 스칼라 h 는 불변.
torch 만으로 구현 (torch_geometric 불필요).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.equivariant.physics_embedded.physics_embedded_gnn import (
    ...     EGNN,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal((10, 3)).astype(np.float32)
    >>> h = rng.standard_normal((10, 4)).astype(np.float32)
    >>> model = EGNN(feat_dim=4, n_layers=2, hidden=16)
    >>> x_new, h_new = model.forward(x, h)
    >>> x_new.shape == (10, 3) and h_new.shape == (10, 4)
    True
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class EGNN:
    """Wrapper — 실제 torch 모듈은 _build() 에서."""

    def __init__(
        self,
        feat_dim: int,
        n_layers: int = 3,
        hidden: int = 32,
        coord_update_scale: float = 1.0,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        self.feat_dim = feat_dim
        self.n_layers = n_layers
        self.hidden = hidden
        self.coord_update_scale = coord_update_scale
        self.device = device
        self.seed = seed
        self._model: Any = None
        self._device: Any = None

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

        F, H, C = self.feat_dim, self.hidden, self.coord_update_scale
        n_layers = self.n_layers

        class _EGNNLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # φ_e: (h_i, h_j, ||r||²) → hidden
                self.edge_mlp = nn.Sequential(
                    nn.Linear(2 * F + 1, H), nn.SiLU(),
                    nn.Linear(H, H), nn.SiLU(),
                )
                # φ_x: edge feat → scalar (좌표 업데이트 가중)
                self.coord_mlp = nn.Sequential(
                    nn.Linear(H, H), nn.SiLU(),
                    nn.Linear(H, 1),
                )
                # φ_h: (h_i, Σm) → Δh
                self.node_mlp = nn.Sequential(
                    nn.Linear(F + H, H), nn.SiLU(),
                    nn.Linear(H, F),
                )

            def forward(self, x: Any, h: Any) -> tuple[Any, Any]:
                N = x.shape[0]
                # 모든 쌍에 대한 차이
                r = x.unsqueeze(0) - x.unsqueeze(1)    # (N, N, 3)
                r2 = (r ** 2).sum(dim=-1, keepdim=True)  # (N, N, 1)
                h_i = h.unsqueeze(1).expand(-1, N, -1)
                h_j = h.unsqueeze(0).expand(N, -1, -1)
                edge_in = torch.cat([h_i, h_j, r2], dim=-1)
                m_ij = self.edge_mlp(edge_in)  # (N, N, H)

                # 자기 자신 제거
                eye = torch.eye(N, device=x.device).bool()
                mask = (~eye).float().unsqueeze(-1)  # (N, N, 1)

                # 좌표 업데이트
                weights = self.coord_mlp(m_ij) * mask  # (N, N, 1)
                x_update = (weights * r).sum(dim=1) * C  # (N, 3)
                x = x + x_update

                # 스칼라 업데이트
                msg = (m_ij * mask).sum(dim=1)  # (N, H)
                delta_h = self.node_mlp(torch.cat([h, msg], dim=-1))
                h = h + delta_h
                return x, h

        class _EGNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList(map(lambda _: _EGNNLayer(), range(n_layers)))

            def forward(self, x: Any, h: Any) -> tuple[Any, Any]:
                layer_idx = 0
                while layer_idx < len(self.layers):
                    x, h = self.layers[layer_idx](x, h)
                    layer_idx += 1
                return x, h

        return _EGNN()

    def forward(
        self, x: NDArray[np.float64], h: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        import torch

        if self._model is None:
            self._device = self._resolve_device()
            self._model = self._build().to(self._device)

        x_t = torch.tensor(np.asarray(x, dtype=np.float32), device=self._device)
        h_t = torch.tensor(np.asarray(h, dtype=np.float32), device=self._device)
        with torch.no_grad():
            x_out, h_out = self._model(x_t, h_t)
        return x_out.cpu().numpy().astype(np.float64), h_out.cpu().numpy().astype(np.float64)


__all__ = ["EGNN"]
