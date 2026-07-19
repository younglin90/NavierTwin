"""Memory-bounded execution planning for large CFD case sets."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Iterator, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ScaleWorkload:
    """Geometry, time, field, and learning-route size contract."""

    n_cases: int
    n_time_steps: int
    n_points: int
    n_cells: int
    input_components: int
    target_components: int
    route: str = "route1"
    dtype_bytes: int = 4
    edge_factor: float = 6.0

    def __post_init__(self) -> None:
        positive = (
            self.n_cases,
            self.n_time_steps,
            self.n_points,
            self.input_components,
            self.target_components,
            self.dtype_bytes,
        )
        if any(value < 1 for value in positive) or self.n_cells < 0:
            raise ValueError("workload sizes must be positive; n_cells may be zero")
        if self.route not in {"route1", "route2"}:
            raise ValueError("route must be route1 or route2")
        if self.edge_factor <= 0:
            raise ValueError("edge_factor must be positive")

    @property
    def sample_count(self) -> int:
        return self.n_cases * self.n_time_steps


@dataclass(frozen=True, slots=True)
class ScaleBudget:
    """Usable machine resources before NavierTwin safety reserves."""

    ram_bytes: int
    vram_bytes: int = 0
    workers: int = 1
    reserve_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.ram_bytes < 1 or self.vram_bytes < 0 or self.workers < 1:
            raise ValueError("RAM/workers must be positive and VRAM non-negative")
        if not 0 <= self.reserve_fraction < 1:
            raise ValueError("reserve_fraction must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class ScalePlan:
    """Deterministic chunk, batch, MPI, and memory recommendation."""

    feasible: bool
    route: str
    total_data_bytes: int
    usable_ram_bytes: int
    usable_vram_bytes: int
    point_chunk_size: int
    case_batch_size: int
    microbatch_size: int
    gradient_accumulation_steps: int
    mpi_ranks: int
    estimated_host_peak_bytes: int
    estimated_device_peak_bytes: int
    point_chunk_count: int
    case_batch_count: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def available_ram_bytes() -> int:
    """Return currently available RAM without requiring psutil."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        if hasattr(os, "sysconf"):
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return pages * page_size
    raise RuntimeError("available RAM could not be detected")


def _memory_terms(workload: ScaleWorkload) -> tuple[int, int, int]:
    scalar = workload.dtype_bytes
    point_fields = workload.input_components + workload.target_components
    raw_per_sample = workload.n_points * point_fields * scalar
    coordinates_per_case = workload.n_points * 3 * scalar
    topology_per_case = workload.n_cells * 9 * 8
    total = (
        workload.sample_count * raw_per_sample
        + workload.n_cases * (coordinates_per_case + topology_per_case)
    )
    if workload.route == "route1":
        host_per_point = (point_fields + 5) * scalar
        device_per_point = (point_fields + 10) * scalar * 6
    else:
        edges_per_point = workload.edge_factor
        host_per_point = int(
            (point_fields + 6) * scalar + edges_per_point * (2 * 8 + 4 * scalar)
        )
        device_per_point = int(
            (point_fields + 16) * scalar * 6
            + edges_per_point * (2 * 8 + 8 * scalar)
        )
    return int(total), max(1, host_per_point), max(1, device_per_point)


def plan_scale(
    workload: ScaleWorkload,
    budget: ScaleBudget,
    *,
    target_effective_batch: int = 4,
) -> ScalePlan:
    """Compute a conservative bounded-memory execution plan."""
    if target_effective_batch < 1:
        raise ValueError("target_effective_batch must be positive")
    total, host_per_point, device_per_point = _memory_terms(workload)
    usable_ram = int(budget.ram_bytes * (1.0 - budget.reserve_fraction))
    usable_vram = int(budget.vram_bytes * (1.0 - budget.reserve_fraction))
    per_worker_ram = max(1, usable_ram // budget.workers)
    point_chunk = min(
        workload.n_points,
        max(1, per_worker_ram // (host_per_point * 2)),
    )
    case_working_bytes = max(1, point_chunk * host_per_point * 2)
    case_batch = min(
        workload.n_cases,
        max(1, usable_ram // (case_working_bytes * budget.workers)),
    )
    sample_device_bytes = max(1, point_chunk * device_per_point)
    microbatch = (
        min(workload.sample_count, max(1, usable_vram // sample_device_bytes))
        if usable_vram > 0
        else 1
    )
    accumulation = max(1, ceil(target_effective_batch / microbatch))
    mpi_ranks = min(budget.workers, workload.n_cases)
    host_peak = min(
        usable_ram,
        case_batch * case_working_bytes * budget.workers,
    )
    device_peak = min(usable_vram, microbatch * sample_device_bytes)
    warnings: list[str] = []
    if point_chunk < workload.n_points:
        warnings.append("point-wise streaming required")
    if case_batch < workload.n_cases:
        warnings.append("case batching required")
    if budget.vram_bytes == 0:
        warnings.append("CUDA VRAM unavailable; training plan uses CPU fallback")
    elif microbatch == 1 and workload.sample_count > 1:
        warnings.append("VRAM limits training to microbatch 1")
    if workload.route == "route1" and workload.n_points >= 1_000_000:
        warnings.append("common-grid interpolation should use disk-backed chunk cache")
    feasible = point_chunk >= min(1024, workload.n_points)
    if not feasible:
        warnings.append("RAM budget is too small for the minimum point chunk")
    return ScalePlan(
        feasible=feasible,
        route=workload.route,
        total_data_bytes=total,
        usable_ram_bytes=usable_ram,
        usable_vram_bytes=usable_vram,
        point_chunk_size=point_chunk,
        case_batch_size=case_batch,
        microbatch_size=microbatch,
        gradient_accumulation_steps=accumulation,
        mpi_ranks=mpi_ranks,
        estimated_host_peak_bytes=host_peak,
        estimated_device_peak_bytes=device_peak,
        point_chunk_count=ceil(workload.n_points / point_chunk),
        case_batch_count=ceil(workload.n_cases / case_batch),
        warnings=tuple(warnings),
    )


def iter_slices(total: int, chunk_size: int) -> Iterator[slice]:
    """Yield complete, non-overlapping bounded slices."""
    if total < 0 or chunk_size < 1:
        raise ValueError("total must be non-negative and chunk_size positive")
    for start in range(0, total, chunk_size):
        yield slice(start, min(total, start + chunk_size))


def iter_case_batches(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    """Yield views/slices of cases without copying their payloads."""
    for item_slice in iter_slices(len(items), batch_size):
        yield items[item_slice]


__all__ = [
    "ScaleBudget",
    "ScalePlan",
    "ScaleWorkload",
    "available_ram_bytes",
    "iter_case_batches",
    "iter_slices",
    "plan_scale",
]
