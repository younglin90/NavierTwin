"""Tests for the required digital-twin scenario coverage matrix."""

from __future__ import annotations

from naviertwin.core.benchmarks.twin_matrix import (
    assess_scenario,
    build_twin_coverage_report,
    required_scenarios,
    run_twin_scenario_benchmark,
)


def test_required_matrix_has_all_36_product_scenarios() -> None:
    scenarios = required_scenarios()

    assert len(scenarios) == 36
    assert len({item.key for item in scenarios}) == 36
    assert {item.spatial_dimension for item in scenarios} == {1, 2, 3}
    assert {item.temporal_mode for item in scenarios} == {"steady", "unsteady"}
    assert {item.mesh_layout for item in scenarios} == {"uniform", "unstructured"}
    assert len({item.case_layout for item in scenarios}) == 3


def test_every_required_scenario_has_at_least_one_runnable_strategy() -> None:
    report = build_twin_coverage_report()

    assert report["scenario_count"] == 36
    assert report["covered_scenario_count"] == 36
    assert report["coverage_fraction"] == 1.0
    assert report["uncovered_scenarios"] == []


def test_matrix_assesses_all_registered_strategies() -> None:
    report = build_twin_coverage_report()

    assert set(report["strategies"]) == {
        "rom",
        "physics",
        "dynamics",
        "operator",
        "mesh_gnn",
        "gino",
        "mesh_gnn_mp",
        "transolver",
    }
    assert all(len(row["strategies"]) == 8 for row in report["scenarios"])


def test_unsteady_varying_geometry_keeps_honest_strategy_limits() -> None:
    scenario = next(
        item
        for item in required_scenarios()
        if item.key == "3d-unsteady-vg-vc-unstructured"
    )
    assessment = assess_scenario(scenario)
    state_by_strategy = {item.strategy: item.state for item in assessment.strategies}

    assert assessment.covered
    assert state_by_strategy["physics"] == "preprocessing_required"
    assert state_by_strategy["rom"] == "unsupported"
    assert state_by_strategy["dynamics"] == "unsupported"
    assert state_by_strategy["operator"] == "preprocessing_required"


def test_geometry_fno_covers_all_dimensions_and_temporal_modes() -> None:
    for scenario in required_scenarios():
        assessment = assess_scenario(scenario)
        operator = next(
            item for item in assessment.strategies if item.strategy == "operator"
        )
        assert operator.state != "unsupported"


def test_all_36_scenarios_execute_through_canonical_workspace() -> None:
    report = run_twin_scenario_benchmark()

    assert report["scenario_count"] == 36
    assert report["passed_scenario_count"] == 36
    assert report["failed_scenarios"] == []
    assert report["max_abs_error"] <= 1e-12
    assert report["peak_python_bytes"] > 0
    assert all(row["dataset_bytes"] > 0 for row in report["scenarios"])
