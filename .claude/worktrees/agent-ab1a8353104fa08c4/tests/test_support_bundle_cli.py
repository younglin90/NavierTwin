"""Support bundle utility tests."""

from __future__ import annotations

import json
import zipfile
from hashlib import sha256
from pathlib import Path

from naviertwin.utils.workflow.artifact_zip import read_manifest


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _path_sha256(path: Path) -> str:
    return sha256(str(path).encode("utf-8")).hexdigest()


def test_support_bundle_parser_accepts_customer_flags() -> None:
    from naviertwin.main import _build_parser

    args = _build_parser().parse_args(
        [
            "support-bundle",
            "--outdir",
            "/tmp/naviertwin-support",
            "--preflight",
            "case.su2",
            "--acceptance-json",
            "acceptance.json",
            "--acceptance-summary",
            "acceptance.md",
            "--include-optional",
            "--zip",
        ]
    )

    assert args.command == "support-bundle"
    assert args.outdir == "/tmp/naviertwin-support"
    assert args.preflight == "case.su2"
    assert args.acceptance_json == "acceptance.json"
    assert args.acceptance_summary == "acceptance.md"
    assert args.include_optional is True
    assert args.zip_bundle is True


def test_inspect_support_bundle_parser_accepts_customer_flags() -> None:
    from naviertwin.main import _build_parser

    args = _build_parser().parse_args(
        ["inspect-support-bundle", "/tmp/naviertwin-support/support-bundle.zip", "--json"]
    )

    assert args.command == "inspect-support-bundle"
    assert args.path == "/tmp/naviertwin-support/support-bundle.zip"
    assert args.as_json is True


def test_run_support_bundle_outputs_json(monkeypatch, capsys, tmp_path) -> None:
    import naviertwin.utils.support_bundle as support_bundle
    from naviertwin.main import _run_support_bundle

    def fake_build_support_bundle(
        outdir: str | Path,
        preflight: str | Path | None = None,
        include_optional: bool = False,
        zip_bundle: bool = False,
        acceptance_json: str | Path | None = None,
        acceptance_summary: str | Path | None = None,
    ) -> dict[str, object]:
        return {
            "status": "ok",
            "version": "test",
            "files": ["doctor.json", "metadata.json"],
            "inputs": {
                "outdir": str(outdir),
                "preflight": None if preflight is None else str(preflight),
                "include_optional": include_optional,
                "zip_bundle": zip_bundle,
                "acceptance_json": None if acceptance_json is None else str(acceptance_json),
                "acceptance_summary": (
                    None if acceptance_summary is None else str(acceptance_summary)
                ),
            },
            "warnings": [],
            "errors": [],
        }

    monkeypatch.setattr(support_bundle, "build_support_bundle", fake_build_support_bundle)

    code = _run_support_bundle(
        outdir=str(tmp_path / "support"),
        preflight="case.su2",
        include_optional=True,
        zip_bundle=True,
        acceptance_json=str(tmp_path / "acceptance.json"),
        acceptance_summary=str(tmp_path / "acceptance.md"),
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "ok"
    assert payload["inputs"]["preflight"] == "case.su2"
    assert payload["inputs"]["include_optional"] is True
    assert payload["inputs"]["zip_bundle"] is True
    assert payload["inputs"]["acceptance_json"].endswith("acceptance.json")
    assert payload["inputs"]["acceptance_summary"].endswith("acceptance.md")


def test_build_support_bundle_writes_doctor_and_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    from naviertwin.utils import doctor
    from naviertwin.utils.support_bundle import build_support_bundle

    monkeypatch.setattr(
        doctor,
        "build_doctor_report",
        lambda include_optional=False: {
            "status": "ok",
            "version": "x.y.z",
            "checks": [],
            "warnings": [],
            "errors": [],
        },
    )

    outdir = tmp_path / "support"
    metadata = build_support_bundle(outdir, include_optional=True)

    doctor_payload = json.loads((outdir / "doctor.json").read_text(encoding="utf-8"))
    metadata_payload = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))

    assert doctor_payload["status"] == "ok"
    assert metadata == metadata_payload
    assert metadata["status"] == "ok"
    assert metadata["schema_version"] == 2
    assert metadata["version"]
    assert metadata["generated_at"]
    assert metadata["files"] == ["doctor.json", "README.txt", "metadata.json"]
    assert metadata["artifacts"]["doctor.json"] == {
        "bytes": (outdir / "doctor.json").stat().st_size,
        "sha256": _file_sha256(outdir / "doctor.json"),
    }
    assert metadata["artifacts"]["README.txt"] == {
        "bytes": (outdir / "README.txt").stat().st_size,
        "sha256": _file_sha256(outdir / "README.txt"),
    }
    assert "# NavierTwin Support Bundle" in (outdir / "README.txt").read_text(
        encoding="utf-8"
    )
    assert metadata["inputs"] == {
        "preflight": {"provided": False},
        "include_optional": True,
        "acceptance_json": {"provided": False},
        "acceptance_summary": {"provided": False},
    }
    assert metadata["warnings"] == []
    assert metadata["errors"] == []


def test_build_support_bundle_writes_preflight_report_when_available(tmp_path) -> None:
    import pytest

    pytest.importorskip("pyvista", reason="pyvista is required for SU2 preflight fixture")
    from naviertwin.utils.support_bundle import build_support_bundle

    fixture = Path(__file__).parent / "fixtures" / "tiny_square.su2"
    outdir = tmp_path / "support"

    metadata = build_support_bundle(outdir, preflight=fixture)

    preflight_payload = json.loads((outdir / "preflight.json").read_text(encoding="utf-8"))
    metadata_payload = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))

    assert preflight_payload["status"] == "ok"
    assert preflight_payload["summary"]["n_points"] == 4
    assert metadata == metadata_payload
    assert metadata["files"] == ["doctor.json", "preflight.json", "README.txt", "metadata.json"]
    assert set(metadata["artifacts"]) == {"doctor.json", "preflight.json", "README.txt"}
    assert metadata["schema_version"] == 2
    assert metadata["inputs"]["preflight"] == {
        "provided": True,
        "suffix": ".su2",
        "path_sha256": _path_sha256(fixture),
    }


def test_build_support_bundle_records_integrity_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    from naviertwin.utils import doctor
    from naviertwin.utils.support_bundle import build_support_bundle

    monkeypatch.setattr(
        doctor,
        "build_doctor_report",
        lambda include_optional=False: {
            "status": "ok",
            "version": "x.y.z",
            "checks": [{"name": "python_version", "status": "ok"}],
            "warnings": [],
            "errors": [],
        },
    )

    outdir = tmp_path / "support"
    metadata = build_support_bundle(outdir)

    artifacts = metadata["artifacts"]
    assert artifacts == {
        "doctor.json": {
            "bytes": (outdir / "doctor.json").stat().st_size,
            "sha256": _file_sha256(outdir / "doctor.json"),
        },
        "README.txt": {
            "bytes": (outdir / "README.txt").stat().st_size,
            "sha256": _file_sha256(outdir / "README.txt"),
        },
    }
    assert json.loads((outdir / "metadata.json").read_text(encoding="utf-8")) == metadata


def test_build_support_bundle_zip_writes_archive_and_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    from naviertwin.utils import doctor
    from naviertwin.utils.support_bundle import build_support_bundle

    monkeypatch.setattr(
        doctor,
        "build_doctor_report",
        lambda include_optional=False: {
            "status": "ok",
            "version": "x.y.z",
            "checks": [],
            "warnings": [],
            "errors": [],
        },
    )

    outdir = tmp_path / "support"
    metadata = build_support_bundle(outdir, zip_bundle=True)
    zip_path = outdir / "support-bundle.zip"

    assert metadata["zip_path"] == "support-bundle.zip"
    assert metadata["zip_path_sha256"] == _path_sha256(zip_path)
    assert str(zip_path) not in json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert {"doctor.json", "README.txt", "metadata.json", "MANIFEST.json"}.issubset(names)

    manifest = read_manifest(zip_path)
    manifest_map = {str(entry["name"]): entry for entry in manifest}
    for artifact in ["doctor.json", "README.txt", "metadata.json"]:
        data = (outdir / artifact).read_bytes()
        assert manifest_map[artifact]["bytes"] == len(data)
        assert manifest_map[artifact]["sha256"] == sha256(data).hexdigest()


def test_build_support_bundle_includes_acceptance_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    from naviertwin.utils import doctor
    from naviertwin.utils.support_bundle import build_support_bundle

    monkeypatch.setattr(
        doctor,
        "build_doctor_report",
        lambda include_optional=False: {
            "status": "ok",
            "version": "x.y.z",
            "checks": [],
            "warnings": [],
            "errors": [],
        },
    )
    acceptance_json = tmp_path / "source-acceptance.json"
    acceptance_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "package": str(tmp_path / "delivery.zip"),
                "acceptance": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    acceptance_summary = tmp_path / "source-acceptance.md"
    acceptance_summary.write_text("# Acceptance\n\nPASS\n", encoding="utf-8")

    outdir = tmp_path / "support"
    metadata = build_support_bundle(
        outdir,
        acceptance_json=acceptance_json,
        acceptance_summary=acceptance_summary,
        zip_bundle=True,
    )

    bundled_acceptance = json.loads((outdir / "acceptance.json").read_text(encoding="utf-8"))
    assert bundled_acceptance["acceptance"]["passed"] is True
    assert (outdir / "acceptance.md").read_text(encoding="utf-8") == "# Acceptance\n\nPASS\n"
    assert metadata["files"] == [
        "doctor.json",
        "acceptance.json",
        "acceptance.md",
        "README.txt",
        "metadata.json",
    ]
    assert set(metadata["artifacts"]) == {
        "doctor.json",
        "acceptance.json",
        "acceptance.md",
        "README.txt",
    }
    assert metadata["inputs"]["acceptance_json"] == {
        "provided": True,
        "suffix": ".json",
        "path_sha256": _path_sha256(acceptance_json),
    }
    assert metadata["inputs"]["acceptance_summary"] == {
        "provided": True,
        "suffix": ".md",
        "path_sha256": _path_sha256(acceptance_summary),
    }
    assert str(acceptance_json) not in json.dumps(
        metadata["inputs"],
        ensure_ascii=False,
        sort_keys=True,
    )
    readme = (outdir / "README.txt").read_text(encoding="utf-8")
    assert "`acceptance.md`" in readme
    assert "human-readable handoff verdict" in readme
    assert str(acceptance_json) not in readme

    with zipfile.ZipFile(outdir / "support-bundle.zip") as zf:
        names = set(zf.namelist())
    assert {"acceptance.json", "acceptance.md", "README.txt", "MANIFEST.json"}.issubset(names)


def test_inspect_support_bundle_reports_directory_and_zip(tmp_path, monkeypatch) -> None:
    from naviertwin.utils import doctor
    from naviertwin.utils.support_bundle import build_support_bundle, inspect_support_bundle

    monkeypatch.setattr(
        doctor,
        "build_doctor_report",
        lambda include_optional=False: {
            "status": "ok",
            "version": "x.y.z",
            "checks": [],
            "warnings": [],
            "errors": [],
        },
    )

    outdir = tmp_path / "support"
    build_support_bundle(outdir, zip_bundle=True)

    directory_report = inspect_support_bundle(outdir)
    assert directory_report["status"] == "ok"
    assert directory_report["kind"] == "directory"
    assert directory_report["metadata"]["schema_version"] == 2
    assert directory_report["artifacts"]["verified"] is True
    assert directory_report["manifest"]["present"] is False

    zip_report = inspect_support_bundle(outdir / "support-bundle.zip")
    assert zip_report["status"] == "ok"
    assert zip_report["kind"] == "zip"
    assert zip_report["manifest"]["verified"] is True
    assert zip_report["artifacts"]["verified"] is True
    assert "README.txt" in zip_report["metadata"]["files"]


def test_inspect_support_bundle_detects_tampered_zip(tmp_path, monkeypatch) -> None:
    from naviertwin.utils import doctor
    from naviertwin.utils.support_bundle import build_support_bundle, inspect_support_bundle

    monkeypatch.setattr(
        doctor,
        "build_doctor_report",
        lambda include_optional=False: {
            "status": "ok",
            "version": "x.y.z",
            "checks": [],
            "warnings": [],
            "errors": [],
        },
    )

    outdir = tmp_path / "support"
    build_support_bundle(outdir, zip_bundle=True)
    zip_path = outdir / "support-bundle.zip"
    original_entries: dict[str, bytes] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            original_entries[name] = archive.read(name)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, data in original_entries.items():
            archive.writestr(name, b"tampered" if name == "README.txt" else data)

    report = inspect_support_bundle(zip_path)

    assert report["status"] == "error"
    assert report["manifest"]["verified"] is False
    assert any("README.txt" in item for item in report["errors"])


def test_run_inspect_support_bundle_outputs_json(capsys, tmp_path, monkeypatch) -> None:
    from naviertwin.main import _run_inspect_support_bundle
    from naviertwin.utils import doctor
    from naviertwin.utils.support_bundle import build_support_bundle

    monkeypatch.setattr(
        doctor,
        "build_doctor_report",
        lambda include_optional=False: {
            "status": "ok",
            "version": "x.y.z",
            "checks": [],
            "warnings": [],
            "errors": [],
        },
    )

    outdir = tmp_path / "support"
    build_support_bundle(outdir, zip_bundle=True)

    code = _run_inspect_support_bundle(
        path=str(outdir / "support-bundle.zip"),
        as_json=True,
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "ok"
    assert payload["kind"] == "zip"
