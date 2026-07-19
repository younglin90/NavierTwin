"""End-to-end benchmarks using repository CFD format fixtures."""

from __future__ import annotations

from pathlib import Path

from naviertwin.core.benchmarks.format_ingest import benchmark_cfd_fixtures

FIXTURES = Path(__file__).parent / "fixtures"


def test_vtk_cgns_openfoam_ingest_and_roundtrip(tmp_path: Path) -> None:
    report = benchmark_cfd_fixtures(
        {
            "vtk": FIXTURES / "minimal.vtk",
            "cgns": FIXTURES / "synthetic.cgns",
            "openfoam": FIXTURES / "openfoam_case",
        },
        tmp_path,
    )

    assert report["format_count"] == 3
    assert report["passed_format_count"] == 3, report["formats"]
    assert report["failed_formats"] == []
    assert report["max_roundtrip_abs_error"] <= 1e-5
    assert all(item["n_points"] > 0 for item in report["formats"])
    assert all(item["ntwin_bytes"] > 0 for item in report["formats"])
    openfoam = next(
        item for item in report["formats"] if item["format_name"] == "openfoam"
    )
    assert openfoam["boundary_count"] > 0
