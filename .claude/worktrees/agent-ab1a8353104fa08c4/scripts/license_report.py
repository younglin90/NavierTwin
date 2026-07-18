"""Emit a customer-facing license and dependency due-diligence report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 compatibility path
    import tomli as tomllib  # type: ignore[no-redef]


RESTRICTED_DEPENDENCY_PREFIXES = {
    "dedalus",
    "dlrbnicsx",
    "firedrake",
    "flowtorch",
    "foamlib",
    "fluidsimfoam",
    "gmsh",
    "nlopt",
    "openturns",
    "pycgns",
    "pygmo",
    "pypdaf",
    "pymeshlab",
    "rbnicsx",
}
RESTRICTED_ALLOWED_EXTRAS = {"full"}
STABLE_RELEASE_CLASSIFIERS = {
    "Development Status :: 4 - Beta",
    "Development Status :: 5 - Production/Stable",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_name(requirement: str) -> str:
    return (
        requirement.split("[", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .split("<", 1)[0]
        .split(";", 1)[0]
        .strip()
        .lower()
    )


def _status_from_checks(checks: list[dict[str, Any]]) -> str:
    statuses = {str(check.get("status", "error")) for check in checks}
    if "error" in statuses:
        return "error"
    if "warn" in statuses:
        return "warn"
    return "ok"


def _dependency_findings(optional_dependencies: dict[str, list[str]]) -> tuple[list[str], dict[str, Any]]:
    violations: list[str] = []
    isolated: dict[str, list[str]] = {}
    extras: dict[str, dict[str, Any]] = {}

    for extra, dependencies in optional_dependencies.items():
        restricted = sorted(
            dependency
            for dependency in dependencies
            if _normalize_name(dependency) in RESTRICTED_DEPENDENCY_PREFIXES
        )
        extras[extra] = {
            "dependency_count": len(dependencies),
            "restricted_dependencies": restricted,
        }
        if restricted:
            isolated[extra] = restricted
        if extra not in RESTRICTED_ALLOWED_EXTRAS:
            violations.extend(f"{extra}: {dependency}" for dependency in restricted)

    return violations, {"extras": extras, "restricted_isolated_to": sorted(isolated)}


def build_license_report(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Build a JSON-compatible licensing and dependency policy report."""
    root = Path(repo_root) if repo_root is not None else _repo_root()
    pyproject = root / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    optional_dependencies = project.get("optional-dependencies", {})

    classifiers = list(project.get("classifiers", []))
    dependency_violations, dependency_details = _dependency_findings(optional_dependencies)
    metadata_errors: list[str] = []
    if project.get("license") != "MIT":
        metadata_errors.append("project.license must remain MIT")
    if project.get("license-files") != ["LICENSE"]:
        metadata_errors.append("project.license-files must include LICENSE")
    if "Development Status :: 3 - Alpha" in classifiers:
        metadata_errors.append("alpha classifier is not customer-release ready")
    if not any(classifier in STABLE_RELEASE_CLASSIFIERS for classifier in classifiers):
        metadata_errors.append("release classifier must be Beta or Stable")

    checks = [
        {
            "name": "license_metadata",
            "status": "ok" if not metadata_errors else "error",
            "details": {
                "license": project.get("license"),
                "license_files": project.get("license-files", []),
                "classifiers": classifiers,
                "errors": metadata_errors,
            },
        },
        {
            "name": "restricted_dependency_isolation",
            "status": "ok" if not dependency_violations else "error",
            "details": {
                **dependency_details,
                "allowed_extras": sorted(RESTRICTED_ALLOWED_EXTRAS),
                "violations": dependency_violations,
            },
        },
    ]

    return {
        "status": _status_from_checks(checks),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": str(project.get("name", "")),
        "version": str(project.get("version", "")),
        "policy": "mit-core-with-restricted-dependencies-isolated-to-full-extra",
        "checks": checks,
        "warnings": [str(check["name"]) for check in checks if check["status"] == "warn"],
        "errors": [str(check["name"]) for check in checks if check["status"] == "error"],
    }


def report_to_json(report: dict[str, Any]) -> str:
    """Serialize a license report to stable JSON."""
    return json.dumps(report, ensure_ascii=False, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON to stdout")
    parser.add_argument("--output", default=None, metavar="PATH", help="Write JSON report to PATH")
    args = parser.parse_args(argv)

    report = build_license_report()
    json_report = report_to_json(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json_report + "\n", encoding="utf-8")
    print(json_report if args.as_json else f"NavierTwin license report: {report['status']}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
