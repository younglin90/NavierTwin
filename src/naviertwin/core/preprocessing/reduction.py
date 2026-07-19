"""Learning-oriented mesh reduction and memory planning contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


class ReductionMethod(str, Enum):
    """Representation produced for a model backend."""

    NONE = "none"
    UNIFORM_GRID = "uniform_grid"
    MESH_COARSEN = "mesh_coarsen"
    POINT_SAMPLE = "point_sample"


class InterpolationMethod(str, Enum):
    """Value transfer method."""

    LINEAR = "linear"
    NEAREST = "nearest"
    CONSERVATIVE = "conservative"


@dataclass(frozen=True)
class ReductionTarget:
    """User-facing reduction target; exactly one selector is required."""

    retain_fraction: float | None = None
    max_points: int | None = None
    resolution: int | None = None
    max_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        selectors = (
            self.retain_fraction,
            self.max_points,
            self.resolution,
            self.max_memory_bytes,
        )
        if sum(value is not None for value in selectors) != 1:
            raise ValueError("exactly one reduction target selector is required")
        if self.retain_fraction is not None and not 0 < self.retain_fraction <= 1:
            raise ValueError("retain_fraction must be in (0, 1]")
        if self.max_points is not None and self.max_points < 1:
            raise ValueError("max_points must be positive")
        if self.resolution is not None and self.resolution < 2:
            raise ValueError("resolution must be at least 2")
        if self.max_memory_bytes is not None and self.max_memory_bytes < 1:
            raise ValueError("max_memory_bytes must be positive")


@dataclass(frozen=True)
class PreprocessingPlan:
    """Immutable plan recorded with every transformed dataset."""

    method: ReductionMethod
    target: ReductionTarget
    interpolation: InterpolationMethod = InterpolationMethod.LINEAR
    conservative_fields: tuple[str, ...] = ()
    preserve_boundaries: bool = True
    generate_valid_mask: bool = True
    generate_sdf: bool = False
    generate_wall_distance: bool = False
    random_seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.conservative_fields)) != len(self.conservative_fields):
            raise ValueError("conservative_fields must be unique")
        if self.generate_wall_distance and not self.preserve_boundaries:
            raise ValueError("wall distance requires preserved boundaries")


@dataclass(frozen=True)
class MappingRecord:
    """Reproducible source-to-target value transfer artifact."""

    source_mesh_id: str
    target_mesh_id: str
    method: ReductionMethod
    interpolation: InterpolationMethod
    source_points: int
    target_points: int
    valid_mask_uri: str
    weights_uri: str = ""
    remap_floor_rel_l2: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_mesh_id or not self.target_mesh_id:
            raise ValueError("mapping mesh ids must be non-empty")
        if self.source_points < 0 or self.target_points < 0:
            raise ValueError("mapping point counts must be non-negative")
        if not self.valid_mask_uri:
            raise ValueError("mapping must persist a valid mask")
        if self.remap_floor_rel_l2 is not None and self.remap_floor_rel_l2 < 0:
            raise ValueError("remap_floor_rel_l2 must be non-negative")


@dataclass(frozen=True)
class MemoryEstimate:
    """Estimated tensor footprint before optimizer/model overhead."""

    points: int
    snapshots: int
    components: int
    dtype_bytes: int
    tensor_bytes: int
    training_bytes: int

    @property
    def training_gib(self) -> float:
        return self.training_bytes / (1024**3)


def estimate_training_memory(
    *,
    points: int,
    snapshots: int,
    components: int,
    dtype_bytes: int = 4,
    training_multiplier: float = 6.0,
) -> MemoryEstimate:
    """Estimate arrays plus gradients/activations using an explicit multiplier."""

    integers = (points, snapshots, components, dtype_bytes)
    if any(value < 1 for value in integers):
        raise ValueError("memory dimensions must be positive")
    if training_multiplier < 1:
        raise ValueError("training_multiplier must be at least 1")
    tensor_bytes = int(points * snapshots * components * dtype_bytes)
    return MemoryEstimate(
        points=points,
        snapshots=snapshots,
        components=components,
        dtype_bytes=dtype_bytes,
        tensor_bytes=tensor_bytes,
        training_bytes=int(math.ceil(tensor_bytes * training_multiplier)),
    )


def uniform_grid_dimensions(
    bounds: Sequence[float], resolution: int, *, padding_fraction: float = 0.02
) -> tuple[int, int, int]:
    """Match NavierTwin's isotropic background-grid dimension calculation."""

    if len(bounds) != 6:
        raise ValueError("bounds must contain 6 values")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    box = np.asarray(bounds, dtype=np.float64)
    span = box[[1, 3, 5]] - box[[0, 2, 4]]
    span = span + np.where(span > 0, span * (2 * padding_fraction), 0.0)
    longest = float(span.max())
    if longest <= 0:
        raise ValueError("bounds are empty")
    step = longest / resolution
    return tuple(
        max(1, int(round(float(size) / step)) + 1) if size > 0 else 1
        for size in span
    )  # type: ignore[return-value]


def resolution_for_retain_fraction(
    bounds: Sequence[float],
    source_points: int,
    retain_fraction: float,
    *,
    min_resolution: int = 2,
    max_resolution: int = 512,
) -> int:
    """Find the uniform-grid resolution closest to a requested point ratio."""

    if source_points < 1:
        raise ValueError("source_points must be positive")
    if not 0 < retain_fraction <= 1:
        raise ValueError("retain_fraction must be in (0, 1]")
    if min_resolution < 2 or max_resolution < min_resolution:
        raise ValueError("invalid resolution search range")
    target = max(1, int(round(source_points * retain_fraction)))
    low, high = min_resolution, max_resolution
    while low < high:
        middle = (low + high) // 2
        points = math.prod(uniform_grid_dimensions(bounds, middle))
        if points < target:
            low = middle + 1
        else:
            high = middle
    candidates = {max(min_resolution, low - 1), low}
    return min(
        candidates,
        key=lambda value: abs(
            math.prod(uniform_grid_dimensions(bounds, value)) - target
        ),
    )


def apply_validity_mask(
    values: NDArray[Any], valid_mask: NDArray[Any], *, fill_value: float = 0.0
) -> tuple[NDArray[Any], NDArray[np.bool_]]:
    """Fill invalid samples while returning the boolean mask as first-class data."""

    array = np.asarray(values)
    mask = np.asarray(valid_mask, dtype=bool)
    if array.shape[: mask.ndim] != mask.shape:
        raise ValueError("valid mask shape must match leading value dimensions")
    broadcast = mask
    while broadcast.ndim < array.ndim:
        broadcast = broadcast[..., None]
    filled = np.where(broadcast, array, fill_value)
    return filled, mask


__all__ = [
    "InterpolationMethod",
    "MappingRecord",
    "MemoryEstimate",
    "PreprocessingPlan",
    "ReductionMethod",
    "ReductionTarget",
    "apply_validity_mask",
    "estimate_training_memory",
    "resolution_for_retain_fraction",
    "uniform_grid_dimensions",
]
