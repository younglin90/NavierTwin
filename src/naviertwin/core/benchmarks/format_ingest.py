"""Measured ingestion and canonical round-trip checks for real CFD fixtures."""

from __future__ import annotations

import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class FormatIngestRun:
    """One measured reader-to-canonical-to-NTwin result."""

    format_name: str
    source: str
    reader: str
    passed: bool
    elapsed_ms: float
    peak_python_bytes: int
    source_bytes: int
    ntwin_bytes: int
    n_points: int
    n_cells: int
    field_count: int
    boundary_count: int
    roundtrip_max_abs_error: float
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _source_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _arrays(dataset: Any) -> dict[tuple[str, str], np.ndarray]:
    result: dict[tuple[str, str], np.ndarray] = {}
    for name in dataset.field_names:
        if name in dataset.mesh.point_data:
            result[("point", name)] = np.asarray(dataset.mesh.point_data[name])
        elif name in dataset.mesh.cell_data:
            result[("cell", name)] = np.asarray(dataset.mesh.cell_data[name])
    return result


def _max_abs_error(original: Any, restored: Any) -> float:
    errors = [
        float(
            np.max(
                np.abs(
                    np.asarray(original.mesh.points, dtype=np.float64)
                    - np.asarray(restored.mesh.points, dtype=np.float64)
                ),
                initial=0.0,
            )
        )
    ]
    original_arrays = _arrays(original)
    restored_arrays = _arrays(restored)
    if set(original_arrays) != set(restored_arrays):
        missing = sorted(set(original_arrays) ^ set(restored_arrays))
        raise AssertionError(f"field association changed during round-trip: {missing}")
    for key, expected in original_arrays.items():
        actual = restored_arrays[key]
        if expected.shape != actual.shape:
            raise AssertionError(
                f"field shape changed during round-trip: {key}: "
                f"{expected.shape} != {actual.shape}"
            )
        errors.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(expected, dtype=np.float64)
                        - np.asarray(actual, dtype=np.float64)
                    ),
                    initial=0.0,
                )
            )
        )
    return max(errors, default=0.0)


def benchmark_cfd_fixture(
    format_name: str,
    source: Path,
    output_dir: Path,
) -> FormatIngestRun:
    """Read one real fixture and verify canonical and NTwin persistence."""
    started = perf_counter()
    tracemalloc.start()
    try:
        from naviertwin.core.cfd_reader import ReaderFactory
        from naviertwin.core.data_model.workspace import TwinWorkspace
        from naviertwin.core.export.ntwin_format import (
            load_dataset,
            load_embedded_project_manifest,
            save_dataset,
        )

        source = Path(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        reader = ReaderFactory.get_reader(source)
        dataset = reader.read(source)
        workspace = TwinWorkspace()
        project = workspace.load_single_dataset(
            dataset,
            name=f"{format_name} fixture",
            source=str(source),
        )
        output_path = output_dir / f"{format_name}.ntwin"
        save_dataset(dataset, output_path, canonical_project=project)
        restored = load_dataset(output_path)
        embedded = load_embedded_project_manifest(output_path)
        if embedded != project:
            raise AssertionError("embedded canonical manifest changed")
        if restored.n_points != dataset.n_points or restored.n_cells != dataset.n_cells:
            raise AssertionError("mesh size changed during round-trip")
        roundtrip_error = _max_abs_error(dataset, restored)
        if roundtrip_error > 1e-5:
            raise AssertionError(f"round-trip error too large: {roundtrip_error}")
        _, peak = tracemalloc.get_traced_memory()
        return FormatIngestRun(
            format_name=format_name,
            source=str(source),
            reader=type(reader).__name__,
            passed=True,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            peak_python_bytes=int(peak),
            source_bytes=_source_size(source),
            ntwin_bytes=output_path.stat().st_size,
            n_points=dataset.n_points,
            n_cells=dataset.n_cells,
            field_count=len(_arrays(dataset)),
            boundary_count=len(project.boundaries),
            roundtrip_max_abs_error=roundtrip_error,
        )
    except Exception as exc:
        _, peak = tracemalloc.get_traced_memory()
        return FormatIngestRun(
            format_name=format_name,
            source=str(source),
            reader="",
            passed=False,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            peak_python_bytes=int(peak),
            source_bytes=_source_size(source) if Path(source).exists() else 0,
            ntwin_bytes=0,
            n_points=0,
            n_cells=0,
            field_count=0,
            boundary_count=0,
            roundtrip_max_abs_error=float("inf"),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        tracemalloc.stop()


def benchmark_cfd_fixtures(
    fixtures: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Benchmark a deterministic collection of real CFD format fixtures."""
    rows = [
        benchmark_cfd_fixture(name, Path(source), output_dir)
        for name, source in fixtures.items()
    ]
    failures = [item.format_name for item in rows if not item.passed]
    finite_errors = [item.roundtrip_max_abs_error for item in rows if item.passed]
    return {
        "schema_version": "1.0",
        "benchmark_kind": "real-cfd-ingest-roundtrip",
        "format_count": len(rows),
        "passed_format_count": len(rows) - len(failures),
        "failed_formats": failures,
        "max_roundtrip_abs_error": max(finite_errors, default=0.0),
        "total_elapsed_ms": sum(item.elapsed_ms for item in rows),
        "peak_python_bytes": max(
            (item.peak_python_bytes for item in rows), default=0
        ),
        "formats": [item.to_dict() for item in rows],
    }


__all__ = [
    "FormatIngestRun",
    "benchmark_cfd_fixture",
    "benchmark_cfd_fixtures",
]
