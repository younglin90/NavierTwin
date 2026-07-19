"""MPI preprocessing, GPU scheduling, and PyTorch DDP launch primitives."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def partition_indices(count: int, rank: int, size: int) -> tuple[int, ...]:
    """Return deterministic round-robin indices owned by one MPI rank."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if size < 1 or not 0 <= rank < size:
        raise ValueError("rank must be within a positive world size")
    return tuple(range(rank, count, size))


def mpi_map(
    items: Sequence[T],
    worker: Callable[[T], R],
    *,
    comm: Any | None = None,
) -> list[R]:
    """Map preprocessing work across ranks and restore source order on every rank."""

    if comm is None:
        return [worker(item) for item in items]
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    local = [(index, worker(items[index])) for index in partition_indices(len(items), rank, size)]
    chunks = comm.allgather(local)
    merged = [pair for chunk in chunks for pair in chunk]
    merged.sort(key=lambda pair: pair[0])
    if [index for index, _value in merged] != list(range(len(items))):
        raise RuntimeError("MPI preprocessing results are incomplete or duplicated")
    return [value for _index, value in merged]


@dataclass(frozen=True, slots=True)
class GpuDevice:
    index: int
    name: str
    free_bytes: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class GpuRequest:
    job_id: str
    required_bytes: int


@dataclass(frozen=True, slots=True)
class GpuAssignment:
    job_id: str
    device_index: int | None
    accepted: bool
    reason: str


def discover_cuda_devices() -> tuple[GpuDevice, ...]:
    """Read current CUDA free memory without importing torch at module import."""

    try:
        import torch
    except ImportError:
        return ()
    if not torch.cuda.is_available():
        return ()
    devices: list[GpuDevice] = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        devices.append(
            GpuDevice(
                index=index,
                name=str(torch.cuda.get_device_name(index)),
                free_bytes=int(free_bytes),
                total_bytes=int(total_bytes),
            )
        )
    return tuple(devices)


def schedule_gpu_requests(
    requests: Sequence[GpuRequest],
    devices: Sequence[GpuDevice],
    *,
    reserve_fraction: float = 0.15,
) -> tuple[GpuAssignment, ...]:
    """Best-fit decreasing assignment using remaining memory on each GPU."""

    if not 0 <= reserve_fraction < 1:
        raise ValueError("reserve_fraction must be in [0, 1)")
    remaining = {
        device.index: int(device.free_bytes * (1.0 - reserve_fraction))
        for device in devices
    }
    assigned: dict[str, GpuAssignment] = {}
    for request in sorted(requests, key=lambda item: item.required_bytes, reverse=True):
        if not request.job_id or request.required_bytes < 1:
            raise ValueError("GPU requests require a job_id and positive bytes")
        candidates = [
            (capacity - request.required_bytes, index)
            for index, capacity in remaining.items()
            if capacity >= request.required_bytes
        ]
        if not candidates:
            assigned[request.job_id] = GpuAssignment(
                request.job_id,
                None,
                False,
                "no GPU has sufficient free memory after reserve",
            )
            continue
        _slack, selected = min(candidates)
        remaining[selected] -= request.required_bytes
        assigned[request.job_id] = GpuAssignment(
            request.job_id,
            selected,
            True,
            f"assigned to cuda:{selected}",
        )
    return tuple(assigned[request.job_id] for request in requests)


@dataclass(frozen=True, slots=True)
class DDPLaunchSpec:
    entrypoint: Path
    entrypoint_args: tuple[str, ...] = ()
    nproc_per_node: int = 1
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    def __post_init__(self) -> None:
        if self.nproc_per_node < 1 or self.nnodes < 1:
            raise ValueError("DDP process and node counts must be positive")
        if not 0 <= self.node_rank < self.nnodes:
            raise ValueError("node_rank must be within nnodes")
        if not 1 <= self.master_port <= 65535:
            raise ValueError("master_port must be in [1, 65535]")


def build_torchrun_command(spec: DDPLaunchSpec) -> list[str]:
    """Build a shell-free ``torch.distributed.run`` command."""

    entrypoint = Path(spec.entrypoint).expanduser().resolve()
    if not entrypoint.is_file():
        raise FileNotFoundError(f"DDP entrypoint not found: {entrypoint}")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(spec.nproc_per_node),
        "--nnodes",
        str(spec.nnodes),
        "--node-rank",
        str(spec.node_rank),
        "--master-addr",
        spec.master_addr,
        "--master-port",
        str(spec.master_port),
        str(entrypoint),
        *spec.entrypoint_args,
    ]


def launch_ddp(
    spec: DDPLaunchSpec,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Launch a blocking DDP job with inherited logs and no shell."""

    launch_env = os.environ.copy()
    launch_env.update(env or {})
    return subprocess.run(
        build_torchrun_command(spec),
        cwd=None if cwd is None else Path(cwd),
        env=launch_env,
        text=True,
        check=False,
    )


__all__ = [
    "DDPLaunchSpec",
    "GpuAssignment",
    "GpuDevice",
    "GpuRequest",
    "build_torchrun_command",
    "discover_cuda_devices",
    "launch_ddp",
    "mpi_map",
    "partition_indices",
    "schedule_gpu_requests",
]
