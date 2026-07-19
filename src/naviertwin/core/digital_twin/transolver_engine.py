"""Transolver case-set engine using the mesh-native prediction contract."""

from __future__ import annotations

from typing import Any

from naviertwin.core.digital_twin.mesh_gnn_engine import MeshGNNTwinEngine


class TransolverTwinEngine(MeshGNNTwinEngine):
    """Mesh-native engine whose operator uses learned physics slices."""

    def __init__(self, operator: Any, **kwargs: Any) -> None:
        super().__init__(operator, **kwargs)
        self.reducer_type = "transolver"
        self.surrogate_type = "physics_attention"
        self.model_type = "transolver"
        self.training_metadata.update(
            reducer="transolver",
            surrogate="physics_attention",
            architecture="physics_attention_slices",
        )


__all__ = ["TransolverTwinEngine"]
