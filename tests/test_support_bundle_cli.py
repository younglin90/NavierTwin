"""Support bundle utility tests."""

from __future__ import annotations

import json
import zipfile
from hashlib import sha256
from pathlib import Path

from naviertwin.utils.workflow.artifact_zip import read_manifest


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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
    assert metadata["version"]
    assert metadata["generated_at"]
    assert metadata["files"] == ["doctor.json", "metadata.json"]
    assert metadata["artifacts"]["doctor.json"] == {
        "bytes": (outdir / "doctor.json").stat().st_size,
        "sha256": _file_sha256(outdir / "doctor.json"),
    }
    assert metadata["inputs"] == {
        "preflight": None,
        "include_optional": True,
        "acceptance_json": None,
        "acceptance_summary": None,
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
    assert metadata["files"] == ["doctor.json", "preflight.json", "metadata.json"]
    assert set(metadata["artifacts"]) == {"doctor.json", "preflight.json"}
    assert metadata["inputs"]["preflight"] == str(fixture)


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
        }
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

    assert metadata["zip_path"] == str(zip_path)
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert {"doctor.json", "metadata.json", "MANIFEST.json"}.issubset(names)

    manifest = read_manifest(zip_path)
    manifest_map = {str(entry["name"]): entry for entry in manifest}
    for artifact in ["doctor.json", "metadata.json"]:
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
        "metadata.json",
    ]
    assert set(metadata["artifacts"]) == {"doctor.json", "acceptance.json", "acceptance.md"}
    assert metadata["inputs"]["acceptance_json"] == str(acceptance_json)
    assert metadata["inputs"]["acceptance_summary"] == str(acceptance_summary)

    with zipfile.ZipFile(outdir / "support-bundle.zip") as zf:
        names = set(zf.namelist())
    assert {"acceptance.json", "acceptance.md", "MANIFEST.json"}.issubset(names)
