"""Dataset readiness checks before CFD import, training, or ROM reduction."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from naviertwin.core.cfd_reader import ReaderFactory
from naviertwin.core.cfd_reader.base import CFDDataset


def _check(
    name: str,
    status: str,
    *,
    details: dict[str, Any] | None = None,
    code: str | None = None,
    message: str | None = None,
    remediation: str | None = None,
) -> dict[str, Any]:
    """Build a preflight check with stable remediation fields on non-ok status."""
    payload: dict[str, Any] = {
        "name": name,
        "status": status,
        "details": details or {},
    }
    if status in {"warn", "error"}:
        payload.update(
            {
                "code": code or name.upper(),
                "message": message or name.replace("_", " "),
                "remediation": remediation
                or "Review the check details, fix the input data, and rerun preflight.",
            }
        )
    return payload


def _final_status(checks: list[dict[str, Any]]) -> str:
    statuses = set(map(lambda check: str(check.get("status", "error")), checks))
    if "error" in statuses:
        return "error"
    if "warn" in statuses:
        return "warn"
    return "ok"


def _readiness_score(checks: list[dict[str, Any]]) -> int:
    if not checks:
        return 0
    ok_count = list(map(lambda check: check.get("status"), checks)).count("ok")
    return round(100 * ok_count / len(checks))


def _message_list(checks: list[dict[str, Any]], status: str) -> list[str]:
    return list(
        map(
            lambda check: str(check.get("name", "unknown")),
            filter(lambda check: check.get("status") == status, checks),
        )
    )


def _field_array(dataset: CFDDataset, field_name: str) -> Any | None:
    mesh = dataset.mesh
    if field_name in mesh.point_data:
        return mesh.point_data[field_name]
    if field_name in mesh.cell_data:
        return mesh.cell_data[field_name]
    return None


def _field_location(dataset: CFDDataset, field_name: str) -> str:
    mesh = dataset.mesh
    if field_name in mesh.point_data:
        return "point"
    if field_name in mesh.cell_data:
        return "cell"
    return "unknown"


def _field_reports(dataset: CFDDataset, checks: list[dict[str, Any]]) -> dict[str, Any]:
    field_sanity = import_module("naviertwin.core.validation.field_sanity")

    fields: dict[str, Any] = {}
    field_names = list(dataset.field_names)
    field_idx = 0
    while field_idx < len(field_names):
        field_name = field_names[field_idx]
        values = _field_array(dataset, field_name)
        if values is None:
            checks.append(
                _check(
                    f"field:{field_name}",
                    "warn",
                    details={"reason": "field name is not present in point_data or cell_data"},
                    code="FIELD_LOCATION_MISSING",
                    message=f"Field {field_name!r} is listed but not present in mesh arrays.",
                    remediation=(
                        "Regenerate or export the case so every listed field exists in "
                        "point_data or cell_data."
                    ),
                )
            )
            fields[field_name] = {"location": "unknown", "sanity": None}
            field_idx += 1
            continue
        try:
            sanity = field_sanity.field_sanity_check(values)
        except (TypeError, ValueError) as exc:
            checks.append(
                _check(
                    f"field:{field_name}",
                    "error",
                    details={"reason": str(exc)},
                    code="FIELD_SANITY_FAILED",
                    message=f"Field {field_name!r} failed numeric sanity checks.",
                    remediation=(
                        "Convert the field to a finite numeric array before training or ROM "
                        "reduction."
                    ),
                )
            )
            fields[field_name] = {"location": _field_location(dataset, field_name), "sanity": None}
            field_idx += 1
            continue

        status = "ok" if sanity.get("all_finite") is True else "error"
        details = {
            "location": _field_location(dataset, field_name),
            "n_nan": sanity.get("n_nan", 0),
            "n_inf": sanity.get("n_inf", 0),
            "size": sanity.get("size", 0),
        }
        checks.append(
            _check(
                f"field:{field_name}",
                status,
                details=details,
                code="FIELD_NON_FINITE",
                message=f"Field {field_name!r} contains NaN or infinite values.",
                remediation=(
                    "Clean or interpolate non-finite field values before running the pipeline."
                ),
            )
        )
        fields[field_name] = {
            "location": _field_location(dataset, field_name),
            "sanity": sanity,
        }
        field_idx += 1
    return fields


def _base_report(path: Path) -> dict[str, Any]:
    return {
        "status": "error",
        "path": str(path),
        "format": path.suffix.lower(),
        "reader": "",
        "readiness_score": 0,
        "checks": [],
        "summary": {},
        "fields": {},
        "warnings": [],
        "errors": [],
    }


def _json_safe(value: Any) -> Any:
    """Convert arrays and paths in reader metadata to compact JSON-safe values."""
    if isinstance(value, dict):
        return dict(map(lambda item: (str(item[0]), _json_safe(item[1])), value.items()))
    if isinstance(value, (list, tuple)):
        return list(map(_json_safe, value))
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _finalize(report: dict[str, Any]) -> dict[str, Any]:
    checks = report["checks"]
    report["status"] = _final_status(checks)
    report["readiness_score"] = _readiness_score(checks)
    report["warnings"] = _message_list(checks, "warn")
    report["errors"] = _message_list(checks, "error")
    return report


def build_dataset_preflight_report(path: str | Path) -> dict[str, Any]:
    """Build a JSON-compatible readiness report describing a CFD input path."""
    input_path = Path(path)
    report = _base_report(input_path)
    checks: list[dict[str, Any]] = report["checks"]

    if not input_path.exists():
        checks.append(
            _check(
                "path_exists",
                "error",
                details={"reason": "path does not exist"},
                code="INPUT_PATH_MISSING",
                message=f"Input path does not exist: {input_path}",
                remediation=(
                    "Check the file or case directory path, then rerun preflight with an "
                    "existing CFD input."
                ),
            )
        )
        return _finalize(report)
    checks.append(_check("path_exists", "ok"))

    try:
        reader = ReaderFactory.get_reader(input_path)
    except (FileNotFoundError, ValueError) as exc:
        checks.append(
            _check(
                "supported_format",
                "error",
                details={"reason": str(exc)},
                code="UNSUPPORTED_FORMAT",
                message=f"Input format is not supported: {input_path.suffix.lower() or 'directory'}",
                remediation=(
                    "Use one of the supported CFD formats or convert the dataset before import."
                ),
            )
        )
        return _finalize(report)

    report["reader"] = type(reader).__name__
    checks.append(_check("supported_format", "ok", details={"reader": report["reader"]}))

    try:
        dataset = reader.read(input_path)
    except Exception as exc:  # noqa: BLE001 - preflight must convert reader failures to a report.
        checks.append(
            _check(
                "read_dataset",
                "error",
                details={"reason": str(exc)},
                code="READER_FAILED",
                message=f"{type(reader).__name__} could not read the input dataset.",
                remediation=(
                    "Open the file in the source CFD tool, verify it is complete and not "
                    "corrupted, then export it again."
                ),
            )
        )
        return _finalize(report)

    checks.append(_check("read_dataset", "ok"))
    geometry_ok = dataset.n_points > 0 and dataset.n_cells > 0
    checks.append(
        _check(
            "mesh_geometry",
            "ok" if geometry_ok else "error",
            details={"n_points": dataset.n_points, "n_cells": dataset.n_cells},
            code="EMPTY_MESH",
            message="Mesh has no points or cells.",
            remediation="Export a non-empty volume or surface mesh before running NavierTwin.",
        )
    )
    checks.append(
        _check(
            "field_count",
            "ok" if dataset.field_names else "warn",
            details={"count": len(dataset.field_names)},
            code="NO_FIELDS",
            message="Dataset does not contain any scalar or vector fields.",
            remediation=(
                "Export at least one solution field such as pressure, velocity, or density."
            ),
        )
    )

    report["summary"] = {
        "n_points": dataset.n_points,
        "n_cells": dataset.n_cells,
        "n_time_steps": dataset.n_time_steps,
        "field_names": list(dataset.field_names),
        "metadata": _json_safe(dataset.metadata),
    }
    report["fields"] = _field_reports(dataset, checks)
    return _finalize(report)


def report_to_json(report: dict[str, Any]) -> str:
    """Serialize a preflight report to stable JSON."""
    return json.dumps(report, ensure_ascii=False, sort_keys=True)


def format_preflight_report(report: dict[str, Any]) -> str:
    """Return a compact human-readable preflight summary."""
    lines = [
        f"NavierTwin preflight: {report.get('status', 'unknown')}",
        f"path: {report.get('path', '')}",
        f"reader: {report.get('reader', '')}",
        f"readiness_score: {report.get('readiness_score', 0)}",
    ]
    summary = report.get("summary", {})
    if isinstance(summary, dict) and summary:
        lines.append(
            "mesh: "
            f"points={summary.get('n_points', 0)}, "
            f"cells={summary.get('n_cells', 0)}, "
            f"fields={len(summary.get('field_names', []))}"
        )
    checks = list(report.get("checks", []))
    check_idx = 0
    while check_idx < len(checks):
        check = checks[check_idx]
        if isinstance(check, dict):
            lines.append(f"- {check.get('name', 'unknown')}: {check.get('status', 'unknown')}")
        check_idx += 1
    return "\n".join(lines)


__all__ = [
    "build_dataset_preflight_report",
    "format_preflight_report",
    "report_to_json",
]
