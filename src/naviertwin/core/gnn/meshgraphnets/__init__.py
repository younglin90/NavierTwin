"""MeshGraphNets public API."""

from naviertwin.core.gnn.meshgraphnets.meshgraphnets import MeshGraphNets
from naviertwin.core.gnn.meshgraphnets.rollout_dataset import (
    case_to_rollout_trajectory,
)

__all__ = ["MeshGraphNets", "case_to_rollout_trajectory"]
