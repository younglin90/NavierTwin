"""Graph neural network public API."""

from naviertwin.core.gnn.base import BaseGNN
from naviertwin.core.gnn.gnn_surrogate import GNNSurrogate
from naviertwin.core.gnn.graph_transformer import HAMLET
from naviertwin.core.gnn.meshgraphnets import MeshGraphNets

__all__ = ["BaseGNN", "GNNSurrogate", "HAMLET", "MeshGraphNets"]
