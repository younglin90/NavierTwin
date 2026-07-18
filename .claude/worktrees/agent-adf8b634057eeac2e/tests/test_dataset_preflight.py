"""Dataset preflight readiness report tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyvista", reason="pyvista is required for reader preflight fixtures")

FIXTURES = Path(__file__).parent / "fixtures"
SU2_PATH = FIXTURES / "tiny_square.su2"


def _assert_non_ok_checks_include_remediation_contract(report: dict[str, object]) -> None:
    checks = report["checks"]
    assert isinstance(checks, list)
    non_ok = [
        check
        for check in checks
        if isinstance(check, dict) and check.get("status") in {"warn", "error"}
    ]
    assert non_ok
    for check in non_ok:
        for key in ("code", "message", "remediation"):
            value = check.get(key)
            assert isinstance(value, str)
            assert value.strip()


def test_dataset_preflight_reports_tiny_su2_readiness() -> None:
    from naviertwin.core.validation.dataset_preflight import build_dataset_preflight_report

    report = build_dataset_preflight_report(SU2_PATH)

    assert report["status"] == "ok"
    assert report["reader"] == "SU2Reader"
    assert report["readiness_score"] == 100
    assert report["summary"]["n_points"] == 4
    assert report["summary"]["n_cells"] > 0
    assert "Density" in report["summary"]["field_names"]
    assert report["fields"]["Density"]["sanity"]["all_finite"] is True
    assert report["errors"] == []


def test_dataset_preflight_missing_path_is_error(tmp_path) -> None:
    from naviertwin.core.validation.dataset_preflight import build_dataset_preflight_report

    report = build_dataset_preflight_report(tmp_path / "missing.su2")

    assert report["status"] == "error"
    assert report["readiness_score"] == 0
    assert report["errors"] == ["path_exists"]
    check = report["checks"][0]
    assert check["code"] == "INPUT_PATH_MISSING"
    _assert_non_ok_checks_include_remediation_contract(report)


def test_dataset_preflight_unsupported_format_has_remediation(tmp_path) -> None:
    from naviertwin.core.validation.dataset_preflight import build_dataset_preflight_report

    path = tmp_path / "case.unsupported"
    path.write_text("not a CFD file", encoding="utf-8")

    report = build_dataset_preflight_report(path)

    assert report["status"] == "error"
    assert report["errors"] == ["supported_format"]
    check = report["checks"][-1]
    assert check["name"] == "supported_format"
    assert check["code"] == "UNSUPPORTED_FORMAT"
    _assert_non_ok_checks_include_remediation_contract(report)


def test_dataset_preflight_reader_failure_has_remediation(tmp_path) -> None:
    from naviertwin.core.validation.dataset_preflight import build_dataset_preflight_report

    path = tmp_path / "corrupt.su2"
    path.write_text("not a valid SU2 mesh", encoding="utf-8")

    report = build_dataset_preflight_report(path)

    assert report["status"] == "error"
    assert report["errors"] == ["read_dataset"]
    check = report["checks"][-1]
    assert check["name"] == "read_dataset"
    assert check["code"] == "READER_FAILED"
    _assert_non_ok_checks_include_remediation_contract(report)


def test_preflight_parser_accepts_json_and_output_flags() -> None:
    from naviertwin.main import _build_parser

    args = _build_parser().parse_args(
        ["preflight", str(SU2_PATH), "--json", "--output", "/tmp/preflight.json"]
    )

    assert args.command == "preflight"
    assert args.path == str(SU2_PATH)
    assert args.as_json is True
    assert args.output == "/tmp/preflight.json"


def test_run_preflight_outputs_json(capsys) -> None:
    from naviertwin.main import _run_preflight

    code = _run_preflight(path=str(SU2_PATH), as_json=True)
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "ok"
    assert payload["summary"]["n_points"] == 4
    assert "Pressure" in payload["fields"]


def test_run_preflight_writes_output_file(capsys, tmp_path) -> None:
    from naviertwin.main import _run_preflight

    output = tmp_path / "support" / "preflight.json"

    code = _run_preflight(path=str(SU2_PATH), as_json=True, output=str(output))
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))

    assert code == 0
    assert stdout_payload == file_payload
    assert file_payload["status"] == "ok"
    assert file_payload["summary"]["n_points"] == 4


def test_preflight_cli_stdout_is_json_only() -> None:
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "QT_QPA_PLATFORM": "offscreen",
        "MPLCONFIGDIR": "/tmp/mpl",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "naviertwin.main",
            "preflight",
            str(SU2_PATH),
            "--json",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["status"] == "ok"
    assert "Traceback" not in result.stderr
