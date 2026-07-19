"""Case-set training wrapper for the Transolver-inspired operator."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.gnn.transolver.transolver import Transolver
from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class CaseSetTransolver(BaseOperator):
    """Learn one point-set operator across cases with different mesh sizes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 64,
        n_layers: int = 4,
        n_slices: int = 32,
        n_heads: int = 4,
        max_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.0,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.n_layers = max(1, int(n_layers))
        self.n_slices = int(n_slices)
        self.n_heads = int(n_heads)
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.seed = seed
        self._model: Any = None
        self._device: Any = None
        self._x_mean: NDArray[np.float32] | None = None
        self._x_std: NDArray[np.float32] | None = None
        self._y_mean: NDArray[np.float32] | None = None
        self._y_std: NDArray[np.float32] | None = None
        self.train_losses_: list[float] = []

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self) -> Any:
        import torch

        if self.seed is not None:
            torch.manual_seed(self.seed)
        return Transolver(
            self.in_dim,
            self.out_dim,
            self.hidden,
            self.n_layers,
            self.n_slices,
            self.n_heads,
            self.dropout,
        )

    @staticmethod
    def _validate_graph(graph: dict[str, Any], in_dim: int, out_dim: int) -> None:
        x = np.asarray(graph["x"])
        y = np.asarray(graph["y"])
        if x.ndim != 2 or x.shape[1] != in_dim:
            raise ValueError(f"graph['x'] 는 (N, {in_dim}) 이어야 합니다: {x.shape}")
        if y.ndim != 2 or y.shape != (x.shape[0], out_dim):
            raise ValueError(
                f"graph['y'] 는 (N, {out_dim}) 이어야 합니다: {y.shape} (N={x.shape[0]})"
            )

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch

        graphs = list(dataset["graphs"])
        if not graphs:
            raise ValueError("학습할 그래프가 없습니다.")
        for graph in graphs:
            self._validate_graph(graph, self.in_dim, self.out_dim)
        all_x = np.concatenate([np.asarray(g["x"], dtype=np.float32) for g in graphs])
        all_y = np.concatenate([np.asarray(g["y"], dtype=np.float32) for g in graphs])
        self._x_mean = all_x.mean(axis=0)
        self._x_std = np.where(all_x.std(axis=0) > 0, all_x.std(axis=0), 1.0).astype(np.float32)
        self._y_mean = all_y.mean(axis=0)
        self._y_std = np.where(all_y.std(axis=0) > 0, all_y.std(axis=0), 1.0).astype(np.float32)
        prepared = [
            (
                torch.tensor((np.asarray(g["x"], dtype=np.float32) - self._x_mean) / self._x_std),
                torch.tensor((np.asarray(g["y"], dtype=np.float32) - self._y_mean) / self._y_std),
            )
            for g in graphs
        ]
        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.MSELoss()
        rng = np.random.default_rng(self.seed)
        self.train_losses_ = []
        self._model.train()
        for _epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for index in rng.permutation(len(prepared)):
                x, y = (tensor.to(self._device) for tensor in prepared[index])
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self._model(x), y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            self.train_losses_.append(epoch_loss / len(prepared))
        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info("CaseSetTransolver 학습 완료: %d cases", len(graphs))

    def predict_graph(self, graph: dict[str, Any]) -> NDArray[np.float64]:
        import torch

        self._check_fitted()
        x = np.asarray(graph["x"], dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"graph['x'] 는 (N, {self.in_dim}) 이어야 합니다: {x.shape}")
        assert self._x_mean is not None and self._x_std is not None
        assert self._y_mean is not None and self._y_std is not None
        x = (x - self._x_mean) / self._x_std
        self._model.eval()
        with torch.no_grad():
            out = self._model(torch.tensor(x, device=self._device))
        return out.cpu().numpy().astype(np.float64) * self._y_std + self._y_mean

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        return self.predict_graph(inputs["graph"])

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        model = state.pop("_model", None)
        state.pop("_device", None)
        buffer = io.BytesIO()
        if model is not None:
            import torch

            torch.save(model.state_dict(), buffer)
        state["_model_bytes"] = buffer.getvalue() if model is not None else None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        model_bytes = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        self._model = None
        self._device = None
        if model_bytes is not None:
            import torch

            self._device = self._resolve_device()
            self._model = self._build().to(self._device)
            self._model.load_state_dict(
                torch.load(io.BytesIO(model_bytes), map_location=self._device)
            )
            self._model.eval()

    def _check_fitted(self) -> None:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("CaseSetTransolver.fit() 을 먼저 호출해야 합니다.")


__all__ = ["CaseSetTransolver"]
