"""Asynchronous training orchestration and resource preflight."""

from naviertwin.core.training.distributed import (
    DDPLaunchSpec,
    GpuAssignment,
    GpuDevice,
    GpuRequest,
    build_torchrun_command,
    discover_cuda_devices,
    launch_ddp,
    mpi_map,
    partition_indices,
    schedule_gpu_requests,
)
from naviertwin.core.training.jobs import (
    JobCancelled,
    JobContext,
    JobRecord,
    JobState,
    TrainingJobManager,
)
from naviertwin.core.training.resources import ResourcePreflight, training_preflight
from naviertwin.core.training.scaling import (
    ScaleBudget,
    ScalePlan,
    ScaleWorkload,
    available_ram_bytes,
    iter_case_batches,
    iter_slices,
    plan_scale,
)

__all__ = [
    "JobCancelled",
    "JobContext",
    "JobRecord",
    "JobState",
    "ResourcePreflight",
    "DDPLaunchSpec",
    "GpuAssignment",
    "GpuDevice",
    "GpuRequest",
    "TrainingJobManager",
    "training_preflight",
    "build_torchrun_command",
    "discover_cuda_devices",
    "launch_ddp",
    "mpi_map",
    "partition_indices",
    "schedule_gpu_requests",
    "ScaleBudget",
    "ScalePlan",
    "ScaleWorkload",
    "available_ram_bytes",
    "iter_case_batches",
    "iter_slices",
    "plan_scale",
]
