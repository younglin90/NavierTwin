#!/usr/bin/env python3
"""Write the required CFD digital-twin scenario coverage matrix."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from naviertwin.core.benchmarks.format_ingest import benchmark_cfd_fixtures
from naviertwin.core.benchmarks.twin_matrix import (
    build_twin_coverage_report,
    run_twin_scenario_benchmark,
)


def _format_tsv(report: dict[str, Any]) -> str:
    lines = [
        "scenario\tdimension\ttemporal\tcase_layout\tmesh_layout\tcovered"
        "\trunnable_strategies"
    ]
    for row in report["scenarios"]:
        scenario = row["scenario"]
        lines.append(
            "\t".join(
                [
                    scenario["key"],
                    str(scenario["spatial_dimension"]),
                    scenario["temporal_mode"],
                    scenario["case_layout"],
                    scenario["mesh_layout"],
                    str(row["covered"]).lower(),
                    ",".join(row["runnable_strategies"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute all scenarios plus real VTK/CGNS/OpenFOAM fixture round-trips",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=ROOT / "tests" / "fixtures",
    )
    args = parser.parse_args(argv)

    coverage = build_twin_coverage_report()
    failed = bool(coverage["uncovered_scenarios"])
    report: dict[str, Any] = coverage
    if args.execute:
        if args.format != "json":
            parser.error("--execute currently requires --format json")
        with tempfile.TemporaryDirectory(prefix="naviertwin-benchmark-") as temp_dir:
            scenario_execution = run_twin_scenario_benchmark()
            format_ingest = benchmark_cfd_fixtures(
                {
                    "vtk": args.fixtures_dir / "minimal.vtk",
                    "cgns": args.fixtures_dir / "synthetic.cgns",
                    "openfoam": args.fixtures_dir / "openfoam_case",
                },
                Path(temp_dir),
            )
        failed = failed or bool(scenario_execution["failed_scenarios"])
        failed = failed or bool(format_ingest["failed_formats"])
        report = {
            "schema_version": "1.0",
            "passed": not failed,
            "coverage": coverage,
            "scenario_execution": scenario_execution,
            "format_ingest": format_ingest,
        }
    text = (
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        if args.format == "json"
        else _format_tsv(coverage)
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
