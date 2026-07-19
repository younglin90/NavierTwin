"""Canonical preprocessing contracts and numerical helpers."""

from naviertwin.core.preprocessing.reduction import (
    InterpolationMethod,
    MappingRecord,
    MemoryEstimate,
    PreprocessingPlan,
    ReductionMethod,
    ReductionTarget,
    apply_validity_mask,
    estimate_training_memory,
    resolution_for_retain_fraction,
    uniform_grid_dimensions,
)
from naviertwin.core.preprocessing.temporal_expansion import (
    expand_unsteady_case_snapshots,
)

__all__ = [
    "InterpolationMethod",
    "MappingRecord",
    "MemoryEstimate",
    "PreprocessingPlan",
    "ReductionMethod",
    "ReductionTarget",
    "apply_validity_mask",
    "estimate_training_memory",
    "expand_unsteady_case_snapshots",
    "resolution_for_retain_fraction",
    "uniform_grid_dimensions",
]
