"""Distributed preprocessing, GPU scheduling, and DDP launch tests."""

from __future__ import annotations

import sys

import pytest

from naviertwin.core.training import (
    DDPLaunchSpec,
    GpuDevice,
    GpuRequest,
    build_torchrun_command,
    mpi_map,
    partition_indices,
    schedule_gpu_requests,
)


def test_round_robin_partition_is_complete() -> None:
    chunks = [partition_indices(8, rank, 3) for rank in range(3)]
    assert sorted(index for chunk in chunks for index in chunk) == list(range(8))


class _FakeComm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 2

    def allgather(self, local):
        assert local == [(0, 0), (2, 4)]
        return [local, [(1, 2), (3, 6)]]


def test_mpi_map_restores_source_order() -> None:
    assert mpi_map([0, 1, 2, 3], lambda value: value * 2, comm=_FakeComm()) == [0, 2, 4, 6]


def test_gpu_scheduler_uses_best_fit_and_reports_overflow() -> None:
    gib = 1024**3
    devices = [GpuDevice(0, "large", 8 * gib, 8 * gib), GpuDevice(1, "small", 4 * gib, 4 * gib)]
    requests = [GpuRequest("medium", 3 * gib), GpuRequest("large", 6 * gib), GpuRequest("overflow", 3 * gib)]

    assignments = schedule_gpu_requests(requests, devices, reserve_fraction=0.0)

    assert assignments[0].device_index == 1
    assert assignments[1].device_index == 0
    assert not assignments[2].accepted


def test_torchrun_command_is_shell_free(tmp_path) -> None:
    entrypoint = tmp_path / "train.py"
    entrypoint.write_text("print('ok')\n", encoding="utf-8")
    spec = DDPLaunchSpec(
        entrypoint,
        ("--epochs", "2"),
        nproc_per_node=2,
        nnodes=2,
        node_rank=1,
        master_addr="10.0.0.1",
        master_port=29600,
    )

    command = build_torchrun_command(spec)

    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert command[-3:] == [str(entrypoint), "--epochs", "2"]
    assert "--nproc-per-node" in command


def test_ddp_spec_validates_rank(tmp_path) -> None:
    with pytest.raises(ValueError, match="node_rank"):
        DDPLaunchSpec(tmp_path / "train.py", nnodes=1, node_rank=1)


def test_launch_ddp_cli_dry_run(tmp_path, capsys) -> None:
    from naviertwin.main import _build_parser, _run_ddp_launch

    entrypoint = tmp_path / "train.py"
    entrypoint.write_text("print('ok')\n", encoding="utf-8")
    args = _build_parser().parse_args(
        [
            "launch-ddp",
            "--entrypoint",
            str(entrypoint),
            "--nproc-per-node",
            "2",
            "--dry-run",
            "--",
            "--epochs",
            "4",
        ]
    )

    assert _run_ddp_launch(args) == 0
    assert "torch.distributed.run" in capsys.readouterr().out
