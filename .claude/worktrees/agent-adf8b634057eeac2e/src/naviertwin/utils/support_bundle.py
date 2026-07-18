"""Support bundle builder with customer diagnostics."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from naviertwin import __version__
from naviertwin.utils.secret_redact import redact, redact_object


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(_json_bytes(payload))


def _artifact_integrity(path: Path, expected_payload: dict[str, Any]) -> dict[str, Any]:
    expected = _json_bytes(expected_payload)
    actual = path.read_bytes()
    if actual != expected:
        raise RuntimeError(f"support bundle artifact verification failed: {path.name}")
    return {
        "bytes": len(actual),
        "sha256": sha256(actual).hexdigest(),
    }


def _file_integrity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def _write_text(path: Path, text: str) -> None:
    path.write_text(redact(text), encoding="utf-8")


def _extend_as_strings(target: list[str], values: Any) -> None:
    if not isinstance(values, list):
        return
    value_idx = 0
    while value_idx < len(values):
        target.append(str(values[value_idx]))
        value_idx += 1


def _append_bullets(lines: list[str], values: list[Any]) -> None:
    value_idx = 0
    while value_idx < len(values):
        lines.append(f"- {values[value_idx]}")
        value_idx += 1


def _artifact_paths(output_dir: Path, names: list[str]) -> list[Path]:
    paths: list[Path] = []
    name_idx = 0
    while name_idx < len(names):
        paths.append(output_dir / names[name_idx])
        name_idx += 1
    return paths


def report_to_json(report: dict[str, Any]) -> str:
    """Serialize support bundle metadata to stable JSON."""
    return json.dumps(report, ensure_ascii=False, sort_keys=True)


def _json_from_bytes(data: bytes, *, name: str) -> dict[str, Any]:
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _final_status(statuses: list[str]) -> str:
    if "error" in statuses:
        return "error"
    if "warn" in statuses:
        return "warn"
    return "ok"


def _input_reference(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False}
    path_text = str(Path(path))
    suffix = Path(path).suffix.lower()
    return {
        "provided": True,
        "suffix": suffix or None,
        "path_sha256": sha256(path_text.encode("utf-8")).hexdigest(),
    }


def _support_bundle_readme(metadata: dict[str, Any]) -> str:
    files = metadata.get("files", [])
    inputs = metadata.get("inputs", {})
    artifacts = metadata.get("artifacts", {})
    warnings = metadata.get("warnings", [])
    errors = metadata.get("errors", [])

    lines = [
        "# NavierTwin Support Bundle",
        "",
        f"- Status: {metadata.get('status', 'unknown')}",
        f"- Generated at: {metadata.get('generated_at', 'unknown')}",
        f"- NavierTwin version: {metadata.get('version', 'unknown')}",
        "",
        "## Start Here",
        "",
        "1. Check `README.txt` as this summary.",
        "2. Open `acceptance.md` first when present; it is the human-readable handoff verdict.",
        "3. Open `metadata.json` with the full file list and integrity hashes.",
        "4. Open `doctor.json` and `preflight.json` when runtime or CFD-readiness issues are suspected.",
        "",
        "## Files",
        "",
    ]
    if isinstance(files, list):
        file_idx = 0
        while file_idx < len(files):
            name = files[file_idx]
            if not isinstance(name, str):
                file_idx += 1
                continue
            artifact = artifacts.get(name) if isinstance(artifacts, dict) else None
            suffix = ""
            if isinstance(artifact, dict):
                suffix = f" ({artifact.get('bytes', '?')} bytes)"
            lines.append(f"- `{name}`{suffix}")
            file_idx += 1
    lines.extend(["", "## Inputs", ""])
    if isinstance(inputs, dict):
        input_keys = sorted(inputs)
        key_idx = 0
        while key_idx < len(input_keys):
            key = input_keys[key_idx]
            value = inputs[key]
            if key == "include_optional":
                lines.append(f"- `{key}`: {bool(value)}")
                key_idx += 1
                continue
            if isinstance(value, dict) and "provided" in value:
                status = "provided" if value.get("provided") else "not provided"
                suffix = value.get("suffix")
                suffix_text = f" ({suffix})" if suffix else ""
                lines.append(f"- `{key}`: {status}{suffix_text}")
                key_idx += 1
                continue
            status = "not provided" if value is None else "provided"
            lines.append(f"- `{key}`: {status}")
            key_idx += 1
    lines.extend(["", "## Warnings", ""])
    if isinstance(warnings, list) and warnings:
        _append_bullets(lines, warnings)
    else:
        lines.append("- None")
    lines.extend(["", "## Errors", ""])
    if isinstance(errors, list) and errors:
        _append_bullets(lines, errors)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _verify_artifact_bytes(
    *,
    name: str,
    data: bytes | None,
    expected: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if data is None:
        return [f"missing artifact: {name}"]
    if expected.get("bytes") != len(data):
        errors.append(f"artifact byte mismatch: {name}")
    if expected.get("sha256") != sha256(data).hexdigest():
        errors.append(f"artifact sha256 mismatch: {name}")
    return errors


def _inspect_support_bundle_directory(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    items = list(path.iterdir())
    item_idx = 0
    while item_idx < len(items):
        item = items[item_idx]
        if item.is_file() and item.name != "support-bundle.zip":
            files[item.name] = item.read_bytes()
        item_idx += 1
    return files


def _inspect_support_bundle_zip(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        files = {}
        name_idx = 0
        while name_idx < len(names):
            name = names[name_idx]
            if name != "MANIFEST.json":
                files[name] = archive.read(name)
            name_idx += 1
        manifest_payload = json.loads(archive.read("MANIFEST.json").decode("utf-8"))
    if not isinstance(manifest_payload, list):
        raise ValueError("MANIFEST.json must contain a list")

    manifest_errors: list[str] = []
    seen: set[str] = set()
    entry_idx = 0
    while entry_idx < len(manifest_payload):
        entry = manifest_payload[entry_idx]
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            manifest_errors.append("invalid MANIFEST.json entry")
            entry_idx += 1
            continue
        name = str(entry["name"])
        if name in seen:
            manifest_errors.append(f"duplicate MANIFEST.json entry: {name}")
            entry_idx += 1
            continue
        seen.add(name)
        manifest_errors.extend(
            _verify_artifact_bytes(name=name, data=files.get(name), expected=entry)
        )
        entry_idx += 1
    missing_names = sorted(set(files) - seen)
    name_idx = 0
    while name_idx < len(missing_names):
        name = missing_names[name_idx]
        manifest_errors.append(f"artifact missing from MANIFEST.json: {name}")
        name_idx += 1

    manifest = {
        "present": True,
        "verified": not manifest_errors,
        "entries": len(manifest_payload),
        "errors": manifest_errors,
    }
    return files, manifest


def inspect_support_bundle(path: str | Path) -> dict[str, Any]:
    """Inspect an existing support bundle directory or ZIP without mutating it."""
    source = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    manifest: dict[str, Any] = {"present": False, "verified": None, "entries": 0, "errors": []}

    if source.is_dir():
        kind = "directory"
        files = _inspect_support_bundle_directory(source)
        warnings.append("MANIFEST.json verification is only available with support-bundle.zip")
    elif source.is_file():
        kind = "zip"
        try:
            files, manifest = _inspect_support_bundle_zip(source)
        except KeyError:
            kind = "zip"
            files = {}
            manifest = {
                "present": False,
                "verified": False,
                "entries": 0,
                "errors": ["missing MANIFEST.json"],
            }
    else:
        return {
            "status": "error",
            "kind": "missing",
            "metadata": {"present": False},
            "manifest": manifest,
            "artifacts": {"verified": False, "checked": [], "errors": ["support bundle not found"]},
            "warnings": [],
            "errors": ["support bundle not found"],
        }

    metadata_bytes = files.get("metadata.json")
    if metadata_bytes is None:
        errors.append("missing metadata.json")
        metadata: dict[str, Any] = {}
    else:
        metadata = _json_from_bytes(metadata_bytes, name="metadata.json")

    artifact_errors: list[str] = []
    checked: list[str] = []
    artifacts = metadata.get("artifacts")
    if isinstance(artifacts, dict):
        artifact_items = sorted(artifacts.items())
        artifact_idx = 0
        while artifact_idx < len(artifact_items):
            name, expected = artifact_items[artifact_idx]
            if not isinstance(name, str) or not isinstance(expected, dict):
                artifact_errors.append("invalid metadata artifact entry")
                artifact_idx += 1
                continue
            checked.append(name)
            artifact_errors.extend(
                _verify_artifact_bytes(name=name, data=files.get(name), expected=expected)
            )
            artifact_idx += 1
    elif metadata:
        artifact_errors.append("metadata artifacts must be a mapping")

    declared_files = metadata.get("files", [])
    if isinstance(declared_files, list):
        file_idx = 0
        while file_idx < len(declared_files):
            name = declared_files[file_idx]
            if isinstance(name, str) and name not in files:
                errors.append(f"missing declared file: {name}")
            file_idx += 1

    _extend_as_strings(errors, manifest.get("errors", []))
    errors.extend(artifact_errors)
    metadata_summary = {
        "present": bool(metadata),
        "status": metadata.get("status"),
        "schema_version": metadata.get("schema_version"),
        "version": metadata.get("version"),
        "files": metadata.get("files", []),
        "inputs": metadata.get("inputs", {}),
    }
    return {
        "status": "error" if errors else "ok",
        "kind": kind,
        "metadata": metadata_summary,
        "manifest": manifest,
        "artifacts": {
            "verified": bool(metadata) and not artifact_errors,
            "checked": checked,
            "errors": artifact_errors,
        },
        "warnings": warnings,
        "errors": errors,
    }


def format_support_bundle_inspection(report: dict[str, Any]) -> str:
    """Format support bundle inspection as a short human-readable report."""
    metadata = report.get("metadata", {})
    manifest = report.get("manifest", {})
    artifacts = report.get("artifacts", {})
    lines = [
        f"NavierTwin support bundle: {report.get('status', 'unknown')}",
        f"kind: {report.get('kind', 'unknown')}",
        f"bundle status: {metadata.get('status', 'unknown') if isinstance(metadata, dict) else 'unknown'}",
        f"schema version: {metadata.get('schema_version', 'unknown') if isinstance(metadata, dict) else 'unknown'}",
        f"manifest verified: {manifest.get('verified') if isinstance(manifest, dict) else None}",
        f"artifacts verified: {artifacts.get('verified') if isinstance(artifacts, dict) else None}",
    ]
    errors = report.get("errors", [])
    if isinstance(errors, list) and errors:
        lines.append("errors:")
        _append_bullets(lines, errors)
    warnings = report.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.append("warnings:")
        _append_bullets(lines, warnings)
    return "\n".join(lines)


def build_support_bundle(
    outdir: str | Path,
    preflight: str | Path | None = None,
    include_optional: bool = False,
    zip_bundle: bool = False,
    acceptance_json: str | Path | None = None,
    acceptance_summary: str | Path | None = None,
) -> dict[str, Any]:
    """Build a support bundle with diagnostics, acceptance evidence, and metadata."""
    from naviertwin.utils.doctor import build_doctor_report
    from naviertwin.utils.doctor import report_to_json as doctor_to_json
    from naviertwin.utils.workflow.artifact_zip import zip_artifacts

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[str] = []
    artifacts: dict[str, dict[str, Any]] = {}
    statuses: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    doctor_report = build_doctor_report(include_optional=include_optional)
    doctor_payload = redact_object(json.loads(doctor_to_json(doctor_report)))
    doctor_path = output_dir / "doctor.json"
    _write_json(doctor_path, doctor_payload)
    written_files.append(doctor_path.name)
    artifacts[doctor_path.name] = _artifact_integrity(doctor_path, doctor_payload)
    statuses.append(str(doctor_payload.get("status", "error")))
    _extend_as_strings(warnings, doctor_payload.get("warnings", []))
    _extend_as_strings(errors, doctor_payload.get("errors", []))

    if preflight is not None:
        from naviertwin.core.validation.dataset_preflight import (
            build_dataset_preflight_report,
        )
        from naviertwin.core.validation.dataset_preflight import (
            report_to_json as preflight_to_json,
        )

        preflight_report = build_dataset_preflight_report(preflight)
        preflight_payload = redact_object(json.loads(preflight_to_json(preflight_report)))
        preflight_path = output_dir / "preflight.json"
        _write_json(preflight_path, preflight_payload)
        written_files.append(preflight_path.name)
        artifacts[preflight_path.name] = _artifact_integrity(preflight_path, preflight_payload)
        statuses.append(str(preflight_payload.get("status", "error")))
        _extend_as_strings(warnings, preflight_payload.get("warnings", []))
        _extend_as_strings(errors, preflight_payload.get("errors", []))

    if acceptance_json is not None:
        acceptance_json_source = Path(acceptance_json)
        acceptance_payload = redact_object(
            json.loads(acceptance_json_source.read_text(encoding="utf-8"))
        )
        if not isinstance(acceptance_payload, dict):
            raise ValueError("acceptance JSON artifact must contain a JSON object")
        acceptance_json_path = output_dir / "acceptance.json"
        _write_json(acceptance_json_path, acceptance_payload)
        written_files.append(acceptance_json_path.name)
        artifacts[acceptance_json_path.name] = _artifact_integrity(
            acceptance_json_path,
            acceptance_payload,
        )
        status = str(acceptance_payload.get("status", "ok"))
        if status in {"ok", "warn", "error"}:
            statuses.append(status)
        elif status == "failed":
            statuses.append("error")

    if acceptance_summary is not None:
        acceptance_summary_source = Path(acceptance_summary)
        acceptance_summary_path = output_dir / "acceptance.md"
        acceptance_summary_path.write_text(
            redact(acceptance_summary_source.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        written_files.append(acceptance_summary_path.name)
        artifacts[acceptance_summary_path.name] = _file_integrity(acceptance_summary_path)

    files = [*written_files, "README.txt", "metadata.json"]
    metadata: dict[str, Any] = {
        "status": _final_status(statuses),
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "files": files,
        "artifacts": artifacts,
        "inputs": {
            "preflight": _input_reference(preflight),
            "include_optional": include_optional,
            "acceptance_json": _input_reference(acceptance_json),
            "acceptance_summary": _input_reference(acceptance_summary),
        },
        "warnings": warnings,
        "errors": errors,
    }

    if zip_bundle:
        zip_path = output_dir / "support-bundle.zip"
        metadata["zip_path"] = zip_path.name
        metadata["zip_path_sha256"] = sha256(str(zip_path).encode("utf-8")).hexdigest()
    metadata = redact_object(metadata)

    readme_path = output_dir / "README.txt"
    _write_text(readme_path, _support_bundle_readme(metadata))
    readme_artifacts = metadata.get("artifacts")
    if not isinstance(readme_artifacts, dict):
        raise RuntimeError("support bundle metadata artifacts must be a mapping")
    readme_artifacts[readme_path.name] = _file_integrity(readme_path)

    metadata_path = output_dir / "metadata.json"
    _write_json(metadata_path, metadata)
    _artifact_integrity(metadata_path, metadata)

    if zip_bundle:
        zip_artifacts(_artifact_paths(output_dir, files), zip_path)

    return metadata


__all__ = [
    "build_support_bundle",
    "format_support_bundle_inspection",
    "inspect_support_bundle",
    "report_to_json",
]
