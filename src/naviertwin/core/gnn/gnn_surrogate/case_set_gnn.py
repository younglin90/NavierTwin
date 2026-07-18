"""CaseSetGNN — 케이스별 그래프 목록으로 학습하는 GCN (GNNSurrogate 의 케이스-세트 판).

:class:`~naviertwin.core.gnn.gnn_surrogate.gnn_surrogate.GNNSurrogate` 와의
차이: ``fit`` 이 공유 ``edge_index`` 하나가 아니라 **그래프 dict 목록**
(:func:`~naviertwin.core.gnn.case_graph.case_to_graph` 반환)을 받는다 —
케이스마다 노드 수 N·에지 수 E 가 달라도 된다(형상 가변). GCN 가중치는
그래프 크기와 무관하므로 한 모델이 모든 케이스를 학습/예측한다.

케이스 수가 5~20 수준이라 케이스별 루프(batch=1, 기존 GNNSurrogate 패턴)로
충분하다. 텐서는 케이스별로 device 에 올린다 — 전 케이스 GPU 상주 금지
(대형 메쉬 × 다수 케이스에서 OOM 방지).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

_PYG_MISSING = "torch_geometric 설치 필요: pip install naviertwin[full]"


def _require_pyg() -> Any:
    try:
        import torch_geometric
    except ImportError as exc:
        raise RuntimeError(_PYG_MISSING) from exc
    return torch_geometric


class CaseSetGNN(BaseOperator):
    """케이스-세트 파라메트릭 GCN — 노드 피처(좌표+μ) → 노드 필드 회귀.

    타깃 표준화는 fit 에 받은 그래프(= train 케이스)만으로 내부 계산한다 —
    group split 시 train-only 정규화가 구조적으로 보장된다.

    Attributes:
        train_losses_: epoch 별 케이스 평균 MSE (표준화 y 기준).
        is_fitted: fit 완료 여부.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 64,
        n_layers: int = 4,
        max_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.n_layers = max(2, int(n_layers))
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = seed

        self._model: Any = None
        self._device: Any = None
        self._y_mean: NDArray[np.float32] | None = None
        self._y_std: NDArray[np.float32] | None = None
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

        dims: list[tuple[int, int]] = []
        prev = self.in_dim
        for _ in range(self.n_layers - 1):
            dims.append((prev, self.hidden))
            prev = self.hidden
        dims.append((prev, self.out_dim))

        class _GCN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.convs = nn.ModuleList(GCNConv(a, b) for a, b in dims)

            def forward(self, x: Any, edge_index: Any) -> Any:
                for idx, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if idx < len(self.convs) - 1:
                        x = torch.relu(x)
                return x

        return _GCN()

    @staticmethod
    def _validate_graph(graph: dict[str, Any], in_dim: int, out_dim: int) -> None:
        x = np.asarray(graph["x"])
        y = np.asarray(graph["y"])
        edge = np.asarray(graph["edge_index"])
        if x.ndim != 2 or x.shape[1] != in_dim:
            raise ValueError(f"graph['x'] 는 (N, {in_dim}) 이어야 합니다: {x.shape}")
        if y.ndim != 2 or y.shape[1] != out_dim or y.shape[0] != x.shape[0]:
            raise ValueError(
                f"graph['y'] 는 (N, {out_dim}) 이어야 합니다: {y.shape} (N={x.shape[0]})"
            )
        if edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError(f"edge_index 는 (2, E) 이어야 합니다: {edge.shape}")
        if edge.size and int(edge.max()) >= x.shape[0]:
            raise ValueError(
                f"edge_index 최대값({int(edge.max())})이 노드 수({x.shape[0]})를 넘습니다."
            )

    def fit(self, dataset: dict[str, Any]) -> None:
        """그래프 목록으로 학습한다.

        Args:
            dataset: ``{"graphs": list[dict]}`` — :func:`case_to_graph` 반환
                dict 목록. 케이스마다 N, E 가 달라도 된다.

        Raises:
            ValueError: 그래프가 없거나 shape 이 선언 차원과 다른 경우.
            RuntimeError: torch_geometric 미설치.
        """
        import torch

        graphs = list(dataset["graphs"])
        if not graphs:
            raise ValueError("학습할 그래프가 없습니다.")
        for graph in graphs:
            self._validate_graph(graph, self.in_dim, self.out_dim)

        # 타깃 표준화 — fit 에 받은(=train) 그래프만으로 계산한다.
        all_y = np.concatenate(
            [np.asarray(g["y"], dtype=np.float32) for g in graphs], axis=0
        )
        self._y_mean = all_y.mean(axis=0)
        std = all_y.std(axis=0)
        self._y_std = np.where(std > 0, std, 1.0).astype(np.float32)

        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        optim = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.MSELoss()

        # CPU 텐서로 한 번만 준비 — device 전송은 케이스별 스텝에서(lazy).
        prepared = [
            (
                torch.tensor(np.asarray(g["x"], dtype=np.float32)),
                torch.tensor(
                    (np.asarray(g["y"], dtype=np.float32) - self._y_mean) / self._y_std
                ),
                torch.tensor(np.asarray(g["edge_index"], dtype=np.int64)),
            )
            for g in graphs
        ]

        rng = np.random.default_rng(self.seed)
        self._model.train()
        self.train_losses_ = []
        for _epoch in range(self.max_epochs):
            order = rng.permutation(len(prepared))
            epoch_loss = 0.0
            for i in order:
                x_cpu, y_cpu, edge_cpu = prepared[i]
                x = x_cpu.to(self._device)
                y = y_cpu.to(self._device)
                edge = edge_cpu.to(self._device)
                optim.zero_grad()
                loss = loss_fn(self._model(x, edge), y)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item())
            self.train_losses_.append(epoch_loss / len(prepared))

        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "CaseSetGNN 학습 완료: %d 그래프, loss=%.6g",
            len(graphs),
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict_graph(self, graph: dict[str, Any]) -> NDArray[np.float64]:
        """그래프 하나의 노드별 필드를 예측한다 (물리 단위로 역표준화).

        Args:
            graph: ``x`` (N, in_dim), ``edge_index`` (2, E) 를 담은 dict
                (``y`` 는 없어도 된다 — 예측 전용 그래프).

        Returns:
            (N, out_dim) float64 예측.
        """
        import torch

        self._check_fitted()
        x = np.asarray(graph["x"], dtype=np.float32)
        edge = np.asarray(graph["edge_index"], dtype=np.int64)
        self._model.eval()
        with torch.no_grad():
            out = self._model(
                torch.tensor(x, device=self._device),
                torch.tensor(edge, dtype=torch.long, device=self._device),
            )
        pred = out.cpu().numpy().astype(np.float64)
        assert self._y_mean is not None and self._y_std is not None
        return pred * self._y_std + self._y_mean

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        """BaseOperator 계약 — ``{"graph": dict}`` 를 받아 노드 예측을 돌려준다."""
        return self.predict_graph(inputs["graph"])

    # ------------------------------------------------------------------
    # pickle 지원 — torch 모듈(로컬 클래스) 대신 state_dict 바이트로 직렬화.
    # MeshGNNTwinEngine.save/load (트윈 계약)가 이 경로를 쓴다.
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        model = state.pop("_model", None)
        state.pop("_device", None)
        if model is not None:
            import io

            import torch

            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            state["_model_bytes"] = buffer.getvalue()
        else:
            state["_model_bytes"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        model_bytes = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        self._model = None
        self._device = None
        if model_bytes is not None:
            import io

            import torch

            self._device = self._resolve_device()
            self._model = self._build().to(self._device)
            self._model.load_state_dict(
                torch.load(io.BytesIO(model_bytes), map_location=self._device)
            )
            self._model.eval()

    def _check_fitted(self) -> None:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("CaseSetGNN.fit() 을 먼저 호출해야 합니다.")


__all__ = ["CaseSetGNN"]
