"""Benchmark dataset and validation-reference public API."""

from naviertwin.core.benchmarks.dataset_catalog import (
    generate_burgers_dataset,
    generate_cavity_dataset,
    generate_heat_dataset,
)
from naviertwin.core.benchmarks.format_ingest import (
    FormatIngestRun,
    benchmark_cfd_fixture,
    benchmark_cfd_fixtures,
)
from naviertwin.core.benchmarks.ghia_cavity import ghia_u_centerline
from naviertwin.core.benchmarks.twin_matrix import (
    BenchmarkScenario,
    ScenarioAssessment,
    StrategyAssessment,
    assess_scenario,
    build_twin_coverage_report,
    required_scenarios,
    run_scenario,
    run_twin_scenario_benchmark,
)

__all__ = [
    "generate_burgers_dataset",
    "generate_cavity_dataset",
    "generate_heat_dataset",
    "ghia_u_centerline",
    "FormatIngestRun",
    "BenchmarkScenario",
    "ScenarioAssessment",
    "StrategyAssessment",
    "assess_scenario",
    "build_twin_coverage_report",
    "required_scenarios",
    "run_scenario",
    "run_twin_scenario_benchmark",
    "benchmark_cfd_fixture",
    "benchmark_cfd_fixtures",
]
