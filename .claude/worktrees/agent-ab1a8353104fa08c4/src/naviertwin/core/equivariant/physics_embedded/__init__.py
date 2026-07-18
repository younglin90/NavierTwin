"""Physics-embedded equivariant model public API."""

from naviertwin.core.equivariant.physics_embedded.lie_algebra_no import (
    SO2Canonicalizer,
    SO2EquivariantOperator,
)
from naviertwin.core.equivariant.physics_embedded.physics_embedded_gnn import EGNN

__all__ = ["EGNN", "SO2Canonicalizer", "SO2EquivariantOperator"]
