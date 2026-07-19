"""Coverage matrix for the supported CFD digital-twin data shapes."""

from __future__ import annotations

import tracemalloc
from dataclasses import asdict, dataclass
from itertools import product
from time import perf_counter
from typing import Any

import numpy as np

from naviertwin.core.digital_twin.strategies import DataProfile
from naviertwin.core.digital_twin.strategy_plugins import (
    CapabilityAxes,
    StrategyRegistry,
    default_strategy_registry,
)

CASE_LAYOUTS = (
    "fixed_geometry_varying_conditions",
    "varying_geometry_fixed_condition",
    "varying_geometry_varying_conditions",
)
TEMPORAL_MODES = ("steady", "unsteady")
MESH_LAYOUTS = ("uniform", "unstructured")
STRATEGY_STATES = ("ready", "preprocessing_required", "unsupported")


@dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    """One required combination of geometry, conditions, time, mesh, and dimension."""

    key: str
    case_layout: str
    temporal_mode: str
    mesh_layout: str
    spatial_dimension: int
    n_cases: int = 4
    n_time_steps: int = 1
    n_parameters: int = 1
    n_points: int = 4096

    @property
    def varying_geometry(self) -> bool:
        return self.case_layout != "fixed_geometry_varying_conditions"

    @property
    def profile(self) -> DataProfile:
        return DataProfile(
            n_cases=self.n_cases,
            n_time_steps=self.n_time_steps,
            total_snapshots=self.n_cases * self.n_time_steps,
            identical_mesh=not self.varying_geometry,
            uniform_grid=self.mesh_layout == "uniform",
            n_points=self.n_points,
            dims=self.spatial_dimension,
            n_params=self.n_parameters,
            topological_dim=self.spatial_dimension,
            embedding_dim=3,
        )


@dataclass(frozen=True, slots=True)
class StrategyAssessment:
    """Readiness of one strategy for one benchmark scenario."""

    strategy: str
    state: str
    reason: str
    preprocessing: str
    compute_backend: str


@dataclass(frozen=True, slots=True)
class ScenarioAssessment:
    """All strategy decisions for one scenario."""

    scenario: BenchmarkScenario
    strategies: tuple[StrategyAssessment, ...]

    @property
    def covered(self) -> bool:
        return any(item.state != "unsupported" for item in self.strategies)

    @property
    def runnable_strategies(self) -> tuple[str, ...]:
        return tuple(
            item.strategy for item in self.strategies if item.state != "unsupported"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": asdict(self.scenario),
            "covered": self.covered,
            "runnable_strategies": list(self.runnable_strategies),
            "strategies": [asdict(item) for item in self.strategies],
        }


@dataclass(frozen=True, slots=True)
class ScenarioRun:
    """Measured execution result for one canonical-data scenario."""

    scenario: BenchmarkScenario
    passed: bool
    elapsed_ms: float
    peak_python_bytes: int
    dataset_bytes: int
    case_count: int
    snapshot_count: int
    geometry_count: int
    selected_strategy: str
    max_abs_error: float
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"scenario": asdict(self.scenario), **asdict(self)}


def required_scenarios() -> tuple[BenchmarkScenario, ...]:
    """Return the deterministic 36-scenario product requirement."""
    scenarios: list[BenchmarkScenario] = []
    for case_layout, temporal_mode, mesh_layout, dimension in product(
        CASE_LAYOUTS, TEMPORAL_MODES, MESH_LAYOUTS, (1, 2, 3)
    ):
        n_time_steps = 8 if temporal_mode == "unsteady" else 1
        n_parameters = 0 if case_layout == "varying_geometry_fixed_condition" else 2
        short_layout = {
            "fixed_geometry_varying_conditions": "fg-vc",
            "varying_geometry_fixed_condition": "vg-fc",
            "varying_geometry_varying_conditions": "vg-vc",
        }[case_layout]
        key = f"{dimension}d-{temporal_mode}-{short_layout}-{mesh_layout}"
        scenarios.append(
            BenchmarkScenario(
                key=key,
                case_layout=case_layout,
                temporal_mode=temporal_mode,
                mesh_layout=mesh_layout,
                spatial_dimension=dimension,
                n_time_steps=n_time_steps,
                n_parameters=n_parameters,
            )
        )
    return tuple(scenarios)


def _capability_blocker(
    scenario: BenchmarkScenario, capability: CapabilityAxes
) -> str | None:
    if scenario.spatial_dimension not in capability.spatial_dims:
        return f"spatial dimension {scenario.spatial_dimension}D is unsupported"
    if scenario.temporal_mode == "steady" and not capability.supports_steady:
        return "steady data is unsupported"
    if scenario.temporal_mode == "unsteady" and not capability.supports_unsteady:
        return "unsteady data is unsupported"
    if scenario.varying_geometry and not capability.supports_varying_geometry:
        return "varying geometry is unsupported"
    can_map_unstructured = any(
        token in capability.preprocessing
        for token in ("uniform-grid", "point-cloud", "mesh-graph")
    )
    if (
        scenario.mesh_layout == "unstructured"
        and not capability.supports_unstructured_mesh
        and not can_map_unstructured
    ):
        return "unstructured meshes are unsupported"
    return None


def assess_scenario(
    scenario: BenchmarkScenario,
    registry: StrategyRegistry | None = None,
) -> ScenarioAssessment:
    """Assess one scenario using both declared capability and runtime checker."""
    active_registry = registry or default_strategy_registry()
    runtime_report = active_registry.report(scenario.profile)
    assessments: list[StrategyAssessment] = []
    for key in active_registry.keys():
        registered = active_registry.get(key)
        capability = registered.capability
        blocker = _capability_blocker(scenario, capability)
        decision = runtime_report[key]
        if blocker is not None:
            state = "unsupported"
            reason = blocker
        elif not decision["ok"]:
            state = "unsupported"
            reason = str(decision["reason"])
        else:
            needs_preprocessing = (
                capability.requires_uniform_grid
                and scenario.mesh_layout != "uniform"
            ) or capability.preprocessing not in {"none", "identical-mesh"}
            state = "preprocessing_required" if needs_preprocessing else "ready"
            reason = str(decision["reason"])
        assessments.append(
            StrategyAssessment(
                strategy=key,
                state=state,
                reason=reason,
                preprocessing=capability.preprocessing,
                compute_backend=capability.compute_backend,
            )
        )
    return ScenarioAssessment(scenario=scenario, strategies=tuple(assessments))


def build_twin_coverage_report(
    registry: StrategyRegistry | None = None,
) -> dict[str, Any]:
    """Build a serializable report and aggregate product coverage counts."""
    active_registry = registry or default_strategy_registry()
    rows = [assess_scenario(item, active_registry) for item in required_scenarios()]
    state_counts = {state: 0 for state in STRATEGY_STATES}
    for row in rows:
        for strategy in row.strategies:
            state_counts[strategy.state] += 1
    uncovered = [row.scenario.key for row in rows if not row.covered]
    return {
        "schema_version": "1.0",
        "scenario_count": len(rows),
        "covered_scenario_count": len(rows) - len(uncovered),
        "coverage_fraction": (len(rows) - len(uncovered)) / len(rows),
        "uncovered_scenarios": uncovered,
        "strategy_state_counts": state_counts,
        "strategies": list(active_registry.keys()),
        "scenarios": [row.to_dict() for row in rows],
    }


def _scenario_parameters(scenario: BenchmarkScenario) -> np.ndarray:
    if scenario.n_parameters == 0:
        return np.empty((scenario.n_cases, 0), dtype=np.float64)
    index = np.arange(scenario.n_cases, dtype=np.float64)
    scale = max(1, scenario.n_cases - 1)
    return np.column_stack((index / scale, 0.1 * (index + 1)))


def _scenario_dataset(
    scenario: BenchmarkScenario,
    case_index: int,
    parameters: np.ndarray,
) -> tuple[Any, np.ndarray, int]:
    import pyvista as pv

    dimensions = {
        1: (9, 1, 1),
        2: (7, 6, 1),
        3: (5, 4, 3),
    }[scenario.spatial_dimension]
    varying = scenario.varying_geometry
    stretch = 1.0 + (0.03 * case_index if varying else 0.0)
    grid = pv.ImageData(dimensions=dimensions, spacing=(stretch, 1.0, 1.0))
    mesh = (
        grid.cast_to_unstructured_grid()
        if scenario.mesh_layout == "unstructured"
        else grid
    )
    points = np.asarray(mesh.points, dtype=np.float64)
    base = points[:, : scenario.spatial_dimension].sum(axis=1)
    condition = float(parameters[case_index].sum()) if parameters.shape[1] else 0.0
    times = (
        np.linspace(0.0, 1.0, scenario.n_time_steps, dtype=np.float64)
        if scenario.n_time_steps > 1
        else np.array([0.0], dtype=np.float64)
    )
    expected = np.column_stack(
        [base + condition + float(time_value) for time_value in times]
    )
    mesh.point_data["p"] = expected[:, 0]
    metadata: dict[str, Any] = {
        "topological_dim": scenario.spatial_dimension,
        "scenario": scenario.key,
    }
    if scenario.n_time_steps > 1:
        metadata["time_series_fields"] = {"p": expected.T.copy()}
        metadata["time_series_locations"] = {"p": "point"}

    from naviertwin.core.cfd_reader.base import CFDDataset

    dataset = CFDDataset(
        mesh=mesh,
        time_steps=times.tolist(),
        field_names=["p"],
        metadata=metadata,
    )
    dataset_bytes = int(points.nbytes + expected.nbytes)
    return dataset, expected, dataset_bytes


def run_scenario(
    scenario: BenchmarkScenario,
    registry: StrategyRegistry | None = None,
) -> ScenarioRun:
    """Execute data construction, canonical conversion, and value checks."""
    started = perf_counter()
    tracemalloc.start()
    try:
        from naviertwin.core.data_model.workspace import TwinWorkspace

        parameters = _scenario_parameters(scenario)
        datasets: list[Any] = []
        expected_fields: list[np.ndarray] = []
        dataset_bytes = int(parameters.nbytes)
        for case_index in range(scenario.n_cases):
            dataset, expected, item_bytes = _scenario_dataset(
                scenario, case_index, parameters
            )
            datasets.append(dataset)
            expected_fields.append(expected)
            dataset_bytes += item_bytes

        parameter_names = tuple(
            f"mu_{index}" for index in range(scenario.n_parameters)
        )
        geometry_ids = tuple(
            "geometry-fixed" if not scenario.varying_geometry else f"geometry-{index}"
            for index in range(scenario.n_cases)
        )
        workspace = TwinWorkspace()
        project = workspace.load_case_set(
            datasets,
            parameters,
            parameter_names,
            name=scenario.key,
            geometry_ids=geometry_ids,
        )
        case_set = project.case_sets[0]
        snapshot_count = sum(len(case.snapshots) for case in case_set.cases)
        max_abs_error = 0.0
        for dataset, expected in zip(datasets, expected_fields):
            actual = np.asarray(dataset.extract_field_snapshots("p"))
            max_abs_error = max(
                max_abs_error,
                float(np.max(np.abs(actual - expected), initial=0.0)),
            )

        dimensions = {mesh.topological_dim for mesh in project.meshes}
        expected_geometries = scenario.n_cases if scenario.varying_geometry else 1
        assessment = assess_scenario(scenario, registry)
        selected_strategy = assessment.runnable_strategies[0]
        checks = (
            len(case_set.cases) == scenario.n_cases,
            snapshot_count == scenario.n_cases * scenario.n_time_steps,
            dimensions == {scenario.spatial_dimension},
            len(project.geometries) == expected_geometries,
            max_abs_error <= 1e-12,
            bool(selected_strategy),
        )
        if not all(checks):
            raise AssertionError(f"scenario contract check failed: {checks}")
        _, peak = tracemalloc.get_traced_memory()
        return ScenarioRun(
            scenario=scenario,
            passed=True,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            peak_python_bytes=int(peak),
            dataset_bytes=dataset_bytes,
            case_count=len(case_set.cases),
            snapshot_count=snapshot_count,
            geometry_count=len(project.geometries),
            selected_strategy=selected_strategy,
            max_abs_error=max_abs_error,
        )
    except Exception as exc:
        _, peak = tracemalloc.get_traced_memory()
        return ScenarioRun(
            scenario=scenario,
            passed=False,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            peak_python_bytes=int(peak),
            dataset_bytes=0,
            case_count=0,
            snapshot_count=0,
            geometry_count=0,
            selected_strategy="",
            max_abs_error=float("inf"),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        tracemalloc.stop()


def run_twin_scenario_benchmark(
    registry: StrategyRegistry | None = None,
) -> dict[str, Any]:
    """Run all 36 product scenarios with deterministic synthetic fields."""
    rows = [run_scenario(item, registry) for item in required_scenarios()]
    failures = [item.scenario.key for item in rows if not item.passed]
    finite_errors = [item.max_abs_error for item in rows if item.passed]
    return {
        "schema_version": "1.0",
        "benchmark_kind": "canonical-contract-e2e",
        "scenario_count": len(rows),
        "passed_scenario_count": len(rows) - len(failures),
        "failed_scenarios": failures,
        "max_abs_error": max(finite_errors, default=0.0),
        "total_elapsed_ms": sum(item.elapsed_ms for item in rows),
        "peak_python_bytes": max(
            (item.peak_python_bytes for item in rows), default=0
        ),
        "scenarios": [item.to_dict() for item in rows],
    }


__all__ = [
    "BenchmarkScenario",
    "CASE_LAYOUTS",
    "MESH_LAYOUTS",
    "ScenarioAssessment",
    "ScenarioRun",
    "StrategyAssessment",
    "STRATEGY_STATES",
    "TEMPORAL_MODES",
    "assess_scenario",
    "build_twin_coverage_report",
    "required_scenarios",
    "run_scenario",
    "run_twin_scenario_benchmark",
]
