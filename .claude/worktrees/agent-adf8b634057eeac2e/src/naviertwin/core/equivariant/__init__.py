"""Equivariant model public API."""

from naviertwin.core.equivariant.base import BaseEquivariant
from naviertwin.core.equivariant.group_equiv_fno import C4EquivariantFNO2D
from naviertwin.core.equivariant.physics_embedded import (
    EGNN,
    SO2Canonicalizer,
    SO2EquivariantOperator,
)

__all__ = [
    "BaseEquivariant",
    "C4EquivariantFNO2D",
    "EGNN",
    "SO2Canonicalizer",
    "SO2EquivariantOperator",
]
