"""Customer-facing runtime diagnostics used by support and first-run checks."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from typing import Any

from naviertwin import __version__
from naviertwin.utils.env_info import collect_env
from naviertwin.utils.secret_redact import redact_object

CORE_DEPENDENCIES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "h5py": "h5py",
    "sklearn": "scikit-learn",
}
OPTIONAL_DEPENDENCIES = {
    "PySide6": "PySide6",
    "vtk": "VTK",
    "pyvista": "PyVista",
    "torch": "PyTorch",
    "fastapi": "FastAPI",
    "uvicorn": "uvicorn",
}
VALID_STATUSES = {"ok", "warn", "error"}


def _package_available(module_name: str) -> bool:
    """Return whether a module can be imported without importing it."""
    return importlib.util.find_spec(module_name) is not None


def _check_python_version() -> dict[str, object]:
    required = (3, 10)
    current = sys.version_info[:3]
    ok = current >= required
    parts = []
    part_idx = 0
    while part_idx < len(current):
        parts.append(str(current[part_idx]))
        part_idx += 1
    return {
        "name": "python_version",
        "status": "ok" if ok else "error",
        "details": {
            "current": ".".join(parts),
            "required": ">=3.10",
        },
    }


def _check_dependencies(
    dependencies: dict[str, str],
    *,
    check_name: str,
    missing_status: str,
) -> dict[str, object]:
    packages = {}
    dependency_items = list(dependencies.items())
    dependency_idx = 0
    while dependency_idx < len(dependency_items):
        module_name, display_name = dependency_items[dependency_idx]
        packages[display_name] = _package_available(module_name)
        dependency_idx += 1
    missing = []
    package_items = list(packages.items())
    package_idx = 0
    while package_idx < len(package_items):
        name, available = package_items[package_idx]
        if not available:
            missing.append(name)
        package_idx += 1
    return {
        "name": check_name,
        "status": missing_status if missing else "ok",
        "details": {
            "packages": packages,
            "missing": missing,
        },
    }


def _check_cuda_hint(environment: dict[str, Any]) -> dict[str, object]:
    available = bool(environment.get("cuda_available", False))
    return {
        "name": "cuda",
        "status": "ok" if available else "warn",
        "details": {
            "available": available,
            "devices": environment.get("cuda_devices", []),
        },
    }


def _status_messages(checks: list[dict[str, object]], status: str) -> list[str]:
    messages = []
    check_idx = 0
    while check_idx < len(checks):
        check = checks[check_idx]
        if check.get("status") == status:
            messages.append(str(check.get("name", "unknown")))
        check_idx += 1
    return messages


def _finalize_status(checks: list[dict[str, object]]) -> str:
    statuses = set()
    check_idx = 0
    while check_idx < len(checks):
        statuses.add(str(checks[check_idx].get("status", "error")))
        check_idx += 1
    if "error" in statuses:
        return "error"
    if "warn" in statuses:
        return "warn"
    return "ok"


def _sanitize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Redact secrets while preserving JSON-compatible report structure."""
    return redact_object(report)


def build_doctor_report(*, include_optional: bool = False) -> dict[str, Any]:
    """Build a deterministic diagnostic report used by customer support.

    Args:
        include_optional: Whether to include GUI/API/GPU dependency probes.

    Returns:
        JSON-compatible diagnostic report.
    """
    environment = collect_env()
    checks = [
        _check_python_version(),
        _check_dependencies(
            CORE_DEPENDENCIES,
            check_name="core_dependencies",
            missing_status="warn",
        ),
        _check_cuda_hint(environment),
    ]
    if include_optional:
        checks.append(
            _check_dependencies(
                OPTIONAL_DEPENDENCIES,
                check_name="optional_dependencies",
                missing_status="warn",
            )
        )

    report = {
        "status": _finalize_status(checks),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "environment": environment,
        "checks": checks,
        "warnings": _status_messages(checks, "warn"),
        "errors": _status_messages(checks, "error"),
    }
    return _sanitize_report(report)


def format_doctor_report(report: dict[str, Any]) -> str:
    """Return a compact human-readable diagnostic summary."""
    lines = [
        f"NavierTwin doctor: {report.get('status', 'unknown')}",
        f"version: {report.get('version', 'unknown')}",
    ]
    checks = report.get("checks", [])
    check_idx = 0
    while isinstance(checks, list) and check_idx < len(checks):
        check = checks[check_idx]
        if not isinstance(check, dict):
            check_idx += 1
            continue
        lines.append(f"- {check.get('name', 'unknown')}: {check.get('status', 'unknown')}")
        check_idx += 1
    warnings = report.get("warnings", [])
    errors = report.get("errors", [])
    if warnings:
        lines.append(f"warnings: {_join_as_strings(warnings)}")
    if errors:
        lines.append(f"errors: {_join_as_strings(errors)}")
    return "\n".join(lines)


def _join_as_strings(values: Any) -> str:
    if not isinstance(values, list):
        return str(values)
    items = []
    item_idx = 0
    while item_idx < len(values):
        items.append(str(values[item_idx]))
        item_idx += 1
    return ", ".join(items)


def report_to_json(report: dict[str, Any]) -> str:
    """Serialize a doctor report to stable JSON."""
    return json.dumps(report, ensure_ascii=False, sort_keys=True)


__all__ = [
    "build_doctor_report",
    "format_doctor_report",
    "report_to_json",
]
