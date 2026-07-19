"""Large CFD memory and execution planning tests."""

from __future__ import annotations

import pytest

from naviertwin.core.training import (
    ScaleBudget,
    ScaleWorkload,
    iter_case_batches,
    iter_slices,
    plan_scale,
)


def test_million_point_route2_plan_is_chunked_and_bounded() -> None:
    gib = 1024**3
    workload = ScaleWorkload(
        n_cases=12,
        n_time_steps=40,
        n_points=2_000_000,
        n_cells=1_800_000,
        input_components=6,
        target_components=4,
        route="route2",
    )
    budget = ScaleBudget(ram_bytes=32 * gib, vram_bytes=12 * gib, workers=4)

    plan = plan_scale(workload, budget)

    assert plan.feasible
    assert plan.total_data_bytes > 32 * gib
    assert plan.estimated_host_peak_bytes <= plan.usable_ram_bytes
    assert plan.estimated_device_peak_bytes <= plan.usable_vram_bytes
    assert 1 <= plan.mpi_ranks <= 4
    assert plan.point_chunk_count >= 1


def test_route1_warns_for_disk_backed_million_point_mapping() -> None:
    gib = 1024**3
    plan = plan_scale(
        ScaleWorkload(8, 10, 1_500_000, 1_400_000, 5, 4, route="route1"),
        ScaleBudget(16 * gib, 4 * gib, workers=4),
    )

    assert any("disk-backed" in warning for warning in plan.warnings)


def test_tiny_budget_is_reported_infeasible() -> None:
    plan = plan_scale(
        ScaleWorkload(2, 1, 100_000, 90_000, 8, 4, route="route2"),
        ScaleBudget(1024, 0),
    )

    assert not plan.feasible
    assert plan.point_chunk_size < 1024


def test_slice_and_case_batch_partition_is_complete() -> None:
    slices = list(iter_slices(10, 4))
    assert [(item.start, item.stop) for item in slices] == [(0, 4), (4, 8), (8, 10)]
    assert list(iter_case_batches(tuple(range(7)), 3)) == [(0, 1, 2), (3, 4, 5), (6,)]


def test_scaling_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="route"):
        ScaleWorkload(1, 1, 1, 0, 1, 1, route="invalid")
    with pytest.raises(ValueError, match="chunk_size"):
        list(iter_slices(1, 0))


def test_plan_scale_cli_json(capsys) -> None:
    import json

    from naviertwin.main import _build_parser, _run_scale_plan

    args = _build_parser().parse_args(
        [
            "plan-scale",
            "--cases",
            "8",
            "--time-steps",
            "20",
            "--points",
            "1000000",
            "--cells",
            "900000",
            "--route",
            "route2",
            "--ram-gb",
            "16",
            "--vram-gb",
            "8",
            "--workers",
            "4",
            "--json",
        ]
    )

    assert _run_scale_plan(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["feasible"]
    assert payload["plan"]["mpi_ranks"] == 4
