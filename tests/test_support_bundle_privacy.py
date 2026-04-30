"""Support bundle privacy hardening tests."""

from __future__ import annotations

import json
import zipfile
from hashlib import sha256


def _assert_secret_absent(text: str) -> None:
    assert "secret123" not in text
    assert "Bearer tok.abc.123" not in text
    assert "API_KEY=abcd1234" not in text


def test_support_bundle_redacts_sensitive_values(tmp_path, monkeypatch) -> None:
    from naviertwin.core.validation import dataset_preflight
    from naviertwin.utils import doctor
    from naviertwin.utils.support_bundle import build_support_bundle

    monkeypatch.setenv("NAVIER_TWIN_TEST_TOKEN", "secret123")

    monkeypatch.setattr(
        doctor,
        "build_doctor_report",
        lambda include_optional=False: {
            "status": "ok",
            "version": "x.y.z",
            "environment": {
                "seeded": "NAVIER_TWIN_TEST_TOKEN=secret123",
                "authorization": "Bearer tok.abc.123",
            },
            "checks": [],
            "warnings": ["token=secret123"],
            "errors": [],
        },
    )
    monkeypatch.setattr(
        dataset_preflight,
        "build_dataset_preflight_report",
        lambda path: {
            "status": "ok",
            "path": str(path),
            "checks": [{"name": "ok", "status": "ok", "details": {"api_key": "abcd1234"}}],
            "summary": {"metadata": {"password": "secret123"}},
            "warnings": ["API_KEY=abcd1234"],
            "errors": [],
        },
    )

    outdir = tmp_path / "support"
    acceptance_json = tmp_path / "acceptance_SECRET_TOKEN=secret123.json"
    acceptance_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "package": str(tmp_path / "delivery_SECRET_TOKEN=secret123.zip"),
                "authorization": "Bearer tok.abc.123",
                "prediction": {"api_key": "abcd1234"},
            }
        ),
        encoding="utf-8",
    )
    acceptance_summary = tmp_path / "acceptance_SECRET_TOKEN=secret123.md"
    acceptance_summary.write_text(
        "# Acceptance\n\nPackage: TOKEN=secret123\nAuthorization: Bearer tok.abc.123\n",
        encoding="utf-8",
    )
    metadata = build_support_bundle(
        outdir=outdir,
        preflight=tmp_path / "input_SECRET_TOKEN=secret123.su2",
        acceptance_json=acceptance_json,
        acceptance_summary=acceptance_summary,
        zip_bundle=True,
    )

    doctor_text = (outdir / "doctor.json").read_text(encoding="utf-8")
    preflight_text = (outdir / "preflight.json").read_text(encoding="utf-8")
    acceptance_json_text = (outdir / "acceptance.json").read_text(encoding="utf-8")
    acceptance_md_text = (outdir / "acceptance.md").read_text(encoding="utf-8")
    readme_text = (outdir / "README.txt").read_text(encoding="utf-8")
    metadata_text = (outdir / "metadata.json").read_text(encoding="utf-8")
    _assert_secret_absent(doctor_text)
    _assert_secret_absent(preflight_text)
    _assert_secret_absent(acceptance_json_text)
    _assert_secret_absent(acceptance_md_text)
    _assert_secret_absent(readme_text)
    _assert_secret_absent(metadata_text)

    assert "TOKEN=***REDACTED***" in doctor_text
    assert '"authorization": "***REDACTED***"' in doctor_text
    assert '"authorization": "***REDACTED***"' in acceptance_json_text
    assert "TOKEN=***REDACTED***" in acceptance_md_text

    with zipfile.ZipFile(outdir / "support-bundle.zip") as zf:
        for name in (
            "doctor.json",
            "preflight.json",
            "acceptance.json",
            "acceptance.md",
            "README.txt",
            "metadata.json",
        ):
            text = zf.read(name).decode("utf-8")
            _assert_secret_absent(text)

    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    _assert_secret_absent(encoded)
    assert isinstance(metadata.get("inputs"), dict)
    assert metadata["inputs"]["preflight"] != str(tmp_path / "input_SECRET_TOKEN=secret123.su2")
    assert metadata["inputs"]["acceptance_json"] != str(acceptance_json)
    assert metadata["inputs"]["acceptance_summary"] != str(acceptance_summary)
    assert metadata["schema_version"] == 2
    assert metadata["inputs"]["preflight"] == {
        "provided": True,
        "suffix": ".su2",
        "path_sha256": sha256(
            str(tmp_path / "input_SECRET_TOKEN=secret123.su2").encode("utf-8")
        ).hexdigest(),
    }
    assert metadata["inputs"]["acceptance_json"] == {
        "provided": True,
        "suffix": ".json",
        "path_sha256": sha256(str(acceptance_json).encode("utf-8")).hexdigest(),
    }
    assert metadata["inputs"]["acceptance_summary"] == {
        "provided": True,
        "suffix": ".md",
        "path_sha256": sha256(str(acceptance_summary).encode("utf-8")).hexdigest(),
    }
    assert metadata["zip_path"] == "support-bundle.zip"
    assert metadata["zip_path_sha256"] == sha256(
        str(outdir / "support-bundle.zip").encode("utf-8")
    ).hexdigest()
    assert str(outdir / "support-bundle.zip") not in encoded
