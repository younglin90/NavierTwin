"""Customer diagnostic doctor CLI contract tests."""

from __future__ import annotations

import json


def test_build_doctor_report_schema_minimal(monkeypatch) -> None:
    import naviertwin.utils.doctor as doctor

    monkeypatch.setattr(
        doctor,
        "collect_env",
        lambda: {
            "python": "3.12.3",
            "platform": "Linux",
            "cuda_available": False,
            "cuda_devices": [],
            "diagnostic": "token=abcdef123456",
        },
    )

    report = doctor.build_doctor_report()
    serialized = json.dumps(report, sort_keys=True)

    assert set(report) == {
        "status",
        "timestamp",
        "version",
        "environment",
        "checks",
        "warnings",
        "errors",
    }
    assert report["status"] in {"ok", "warn", "error"}
    assert "abcdef123456" not in serialized
    assert "REDACTED" in serialized
    assert not any(check["name"] == "optional_dependencies" for check in report["checks"])


def test_build_doctor_report_optional_checks_enabled(monkeypatch) -> None:
    import naviertwin.utils.doctor as doctor

    monkeypatch.setattr(
        doctor,
        "collect_env",
        lambda: {
            "python": "3.12.3",
            "platform": "Linux",
            "cuda_available": False,
            "cuda_devices": [],
        },
    )

    report = doctor.build_doctor_report(include_optional=True)

    assert any(check["name"] == "optional_dependencies" for check in report["checks"])


def test_doctor_parser_accepts_json_flag() -> None:
    from naviertwin.main import _build_parser

    args = _build_parser().parse_args(["doctor", "--json", "--include-optional"])

    assert args.command == "doctor"
    assert args.as_json is True
    assert args.include_optional is True
    assert args.output is None


def test_run_doctor_outputs_json(monkeypatch, capsys) -> None:
    import naviertwin.main as main

    monkeypatch.setattr(
        "naviertwin.utils.doctor.build_doctor_report",
        lambda include_optional=False: {
            "status": "ok",
            "timestamp": "2026-04-29T00:00:00+00:00",
            "version": "4.2.58",
            "environment": {"python": "3.12.3"},
            "checks": [{"name": "python_version", "status": "ok", "details": {}}],
            "warnings": [],
            "errors": [],
        },
    )

    code = main._run_doctor(as_json=True, include_optional=False)
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "ok"
    assert payload["checks"][0]["name"] == "python_version"


def test_run_doctor_writes_output_file(monkeypatch, capsys, tmp_path) -> None:
    import naviertwin.main as main

    monkeypatch.setattr(
        "naviertwin.utils.doctor.build_doctor_report",
        lambda include_optional=False: {
            "status": "ok",
            "timestamp": "2026-04-29T00:00:00+00:00",
            "version": "4.2.58",
            "environment": {"diagnostic": "token=***REDACTED***"},
            "checks": [{"name": "secrets_redaction", "status": "ok", "details": {}}],
            "warnings": [],
            "errors": [],
        },
    )
    output = tmp_path / "support" / "doctor.json"

    code = main._run_doctor(as_json=True, include_optional=False, output=str(output))
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))

    assert code == 0
    assert stdout_payload == file_payload
    assert "token=***REDACTED***" in output.read_text(encoding="utf-8")
    assert "abcdef123456" not in output.read_text(encoding="utf-8")
