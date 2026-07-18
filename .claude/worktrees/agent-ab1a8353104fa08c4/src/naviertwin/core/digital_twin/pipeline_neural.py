"""Neural operator / PINN / GNN 기반 NavierTwin 파이프라인 확장.

기존 NavierTwinPipeline (POD + Kriging) 외에 3 가지 신경 파이프라인 제공:

    - NeuralOperatorPipeline: (params, snapshots) → FNO/DeepONet/UNet 직접 매핑
    - PINNPipeline: PDE residual + BC 기반 무데이터 학습
    - GNNPipeline: 비정형 메쉬 node feature → GCN surrogate

모두 `fit/predict/validate/export_report` 공통 API.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.digital_twin.pipeline_neural import NeuralOperatorPipeline
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 32, 1)).astype(np.float32)
    >>> Y = np.sin(X).astype(np.float32)
    >>> pipe = NeuralOperatorPipeline(kind="fno1d", in_ch=1, out_ch=1,
    ...                               modes=4, width=8, n_layers=2, max_epochs=2)
    >>> pipe.fit(X, Y)
    >>> pipe.predict(X[:2]).shape
    (2, 32, 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NeuralState:
    model: Any = None
    train_losses: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class NeuralOperatorPipeline:
    """FNO1D/FNO2D/DeepONet/UNet2D 기반 파이프라인.

    kind 별 필요 파라미터:
        fno1d:    in_ch, out_ch, modes, width, n_layers
        fno2d:    in_ch, out_ch, modes1, modes2, width, n_layers
        unet2d:   in_ch, out_ch, base_ch
        deeponet: branch_in, trunk_in, hidden, latent
    """

    def __init__(self, kind: str = "fno1d", **kwargs: Any) -> None:
        self.kind = kind
        self.kwargs = kwargs
        self.state = NeuralState()

    def _build(self) -> Any:
        if self.kind == "fno1d":
            from naviertwin.core.operator_learning.fno.fno import FNO1D

            return FNO1D(
                in_channels=self.kwargs.get("in_ch", 1),
                out_channels=self.kwargs.get("out_ch", 1),
                modes=self.kwargs.get("modes", 8),
                width=self.kwargs.get("width", 16),
                n_layers=self.kwargs.get("n_layers", 4),
                max_epochs=self.kwargs.get("max_epochs", 100),
                batch_size=self.kwargs.get("batch_size", 16),
                lr=self.kwargs.get("lr", 1e-3),
                device=self.kwargs.get("device", "auto"),
            )
        if self.kind == "fno2d":
            from naviertwin.core.operator_learning.fno.fno import FNO2D

            return FNO2D(
                in_channels=self.kwargs.get("in_ch", 1),
                out_channels=self.kwargs.get("out_ch", 1),
                modes1=self.kwargs.get("modes1", 8),
                modes2=self.kwargs.get("modes2", 8),
                width=self.kwargs.get("width", 16),
                n_layers=self.kwargs.get("n_layers", 4),
                max_epochs=self.kwargs.get("max_epochs", 100),
            )
        if self.kind == "unet2d":
            from naviertwin.core.operator_learning.unet.unet import UNet2D

            return UNet2D(
                in_channels=self.kwargs.get("in_ch", 1),
                out_channels=self.kwargs.get("out_ch", 1),
                base_ch=self.kwargs.get("base_ch", 16),
                max_epochs=self.kwargs.get("max_epochs", 50),
            )
        if self.kind == "deeponet":
            from naviertwin.core.operator_learning.deeponet.deeponet import DeepONet

            return DeepONet(
                branch_in=self.kwargs["branch_in"],
                trunk_in=self.kwargs["trunk_in"],
                hidden=self.kwargs.get("hidden", 64),
                latent=self.kwargs.get("latent", 32),
                max_epochs=self.kwargs.get("max_epochs", 100),
            )
        raise ValueError(f"알 수 없는 kind: {self.kind}")

    def fit(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64] | None = None,
        **extra: Any,
    ) -> None:
        model = self._build()
        if self.kind == "deeponet":
            model.fit({
                "branch_inputs": np.asarray(X, dtype=np.float32),
                "trunk_inputs": np.asarray(extra["trunk_inputs"], dtype=np.float32),
                "outputs": np.asarray(Y, dtype=np.float32),
            })
        else:
            model.fit({"inputs": X, "outputs": Y})
        self.state.model = model
        self.state.train_losses = list(model.train_losses_)
        logger.info(
            "NeuralOperatorPipeline(%s) fit done: final_loss=%.6g",
            self.kind, self.state.train_losses[-1] if self.state.train_losses else 0.0,
        )

    def predict(
        self,
        X: NDArray[np.float64],
        **extra: Any,
    ) -> NDArray[np.float64]:
        if self.state.model is None:
            raise RuntimeError("fit() 먼저 호출")
        if self.kind == "deeponet":
            return self.state.model.predict({
                "branch_inputs": X,
                "trunk_inputs": extra.get("trunk_inputs"),
            })
        return self.state.model.predict({"x": X})

    def validate(
        self,
        X_test: NDArray[np.float64],
        Y_test: NDArray[np.float64],
        **extra: Any,
    ) -> dict[str, float]:
        from naviertwin.core.validation.metrics import compute_all_metrics

        Y_pred = self.predict(X_test, **extra)
        m = compute_all_metrics(np.asarray(Y_test, dtype=np.float64).ravel(),
                                 np.asarray(Y_pred, dtype=np.float64).ravel())
        self.state.metrics = m
        return m

    def export_report(self, path: str | Path, project: str = "NavierTwin NN") -> Path:
        from naviertwin.core.report.generator import ReportGenerator

        data = {
            "project": project,
            "summary": f"{self.kind.upper()} 기반 신경 연산자 파이프라인",
            "metrics": self.state.metrics,
            "model_info": {
                "kind": self.kind,
                **dict(map(lambda item: (item[0], str(item[1])), self.kwargs.items())),
            },
            "notes": (
                f"최종 훈련 loss={self.state.train_losses[-1]:.6g}"
                if self.state.train_losses else ""
            ),
        }
        return ReportGenerator().render_html(data, path)


class PINNPipeline:
    """PDE residual + BC 기반 무데이터 PINN 파이프라인."""

    def __init__(self, in_dim: int = 1, out_dim: int = 1, **kwargs: Any) -> None:
        from naviertwin.core.physnemo.pina_wrapper import PINNSolver

        self.solver = PINNSolver(
            in_dim=in_dim, out_dim=out_dim,
            hidden=kwargs.get("hidden", 32),
            n_layers=kwargs.get("n_layers", 3),
            max_epochs=kwargs.get("max_epochs", 300),
            lr=kwargs.get("lr", 5e-3),
            device=kwargs.get("device", "auto"),
        )

    def fit(
        self,
        residual_fn: Any,
        collocation: NDArray[np.float64],
        boundary: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        self.solver.fit(residual_fn, collocation, boundary)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.solver.predict(x)


class GNNPipeline:
    """GCN surrogate 기반 비정형 메쉬 파이프라인."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs: Any) -> None:
        from naviertwin.core.gnn.gnn_surrogate.gnn_surrogate import GNNSurrogate

        self.model = GNNSurrogate(
            in_dim=in_dim, out_dim=out_dim,
            hidden=kwargs.get("hidden", 32),
            n_layers=kwargs.get("n_layers", 3),
            max_epochs=kwargs.get("max_epochs", 100),
            lr=kwargs.get("lr", 1e-3),
        )

    def fit(
        self,
        node_features: NDArray[np.float64],
        outputs: NDArray[np.float64],
        edge_index: NDArray[np.int64],
    ) -> None:
        self.model.fit({
            "node_features": node_features,
            "outputs": outputs,
            "edge_index": edge_index,
        })

    def predict(
        self,
        x: NDArray[np.float64],
        edge_index: NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        inputs = {"x": x}
        if edge_index is not None:
            inputs["edge_index"] = edge_index
        return self.model.predict(inputs)


__all__ = [
    "NeuralOperatorPipeline",
    "PINNPipeline",
    "GNNPipeline",
]
