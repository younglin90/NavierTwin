"""Round 571 — verification report emitter (CI artifact)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


class TestEmitter:
    def test_pass_path(self, tmp_path) -> None:
        # build inputs
        (tmp_path / "smoke.json").write_text(json.dumps({
            "installer_smoke_pass": True,
            "release_smoke_pass": True,
            "wheel_smoke_pass": True,
            "sdist_smoke_pass": True,
            "smoke_duration_s": 1.0,
        }))
        (tmp_path / "unit.json").write_text(
            json.dumps({"passed": 100, "failed": 0, "exit_code": 0})
        )
        (tmp_path / "coverage.json").write_text(json.dumps({"coverage_pct": 85}))
        (tmp_path / "mms.json").write_text(json.dumps([
            {"name": "x", "observed_p": 2.0, "target": 2.0},
        ]))
        (tmp_path / "vv.json").write_text(json.dumps([
            {"case": "y", "validated": True},
        ]))
        (tmp_path / "drift.json").write_text(json.dumps({"drift_score": 0.05}))
        (tmp_path / "security.json").write_text(json.dumps({"security_findings": 0}))

        script = Path(__file__).resolve().parent.parent / "scripts" / "emit_verification_report.py"
        rc = subprocess.run(
            [sys.executable, str(script), str(tmp_path)],
            capture_output=True, text=True, check=False,
        )
        assert rc.returncode == 0, rc.stderr
        report = json.loads((tmp_path / "verification_report.json").read_text())
        assert report["overall"] == "PASS"
        assert report["layers"]["L0_smoke"] is True
        assert report["details"]["smoke"]["installer_smoke_pass"] is True
        md = (tmp_path / "verification_report.md").read_text()
        assert "PASS" in md

    def test_installer_smoke_failure_fails_l0(self, tmp_path) -> None:
        (tmp_path / "smoke.json").write_text(json.dumps({
            "installer_smoke_pass": False,
            "release_smoke_pass": True,
            "wheel_smoke_pass": True,
            "sdist_smoke_pass": True,
            "smoke_duration_s": 1.0,
        }))
        (tmp_path / "unit.json").write_text(
            json.dumps({"passed": 100, "failed": 0, "exit_code": 0})
        )
        (tmp_path / "coverage.json").write_text(json.dumps({"coverage_pct": 85}))
        (tmp_path / "mms.json").write_text(json.dumps([
            {"name": "x", "observed_p": 2.0, "target": 2.0},
        ]))
        (tmp_path / "vv.json").write_text(json.dumps([
            {"case": "y", "validated": True},
        ]))
        (tmp_path / "drift.json").write_text(json.dumps({"drift_score": 0.05}))
        (tmp_path / "security.json").write_text(json.dumps({"security_findings": 0}))

        script = Path(__file__).resolve().parent.parent / "scripts" / "emit_verification_report.py"
        rc = subprocess.run(
            [sys.executable, str(script), str(tmp_path)],
            capture_output=True, text=True, check=False,
        )

        assert rc.returncode == 1
        report = json.loads((tmp_path / "verification_report.json").read_text())
        assert report["overall"] == "FAIL"
        assert report["layers"]["L0_smoke"] is False
        assert report["details"]["smoke"]["installer_smoke_pass"] is False

    def test_missing_inputs_default_zero(self, tmp_path) -> None:
        script = Path(__file__).resolve().parent.parent / "scripts" / "emit_verification_report.py"
        rc = subprocess.run(
            [sys.executable, str(script), str(tmp_path)],
            capture_output=True, text=True, check=False,
        )
        # all defaults → fails coverage gate (0% < 70%)
        assert rc.returncode == 1
        report = json.loads((tmp_path / "verification_report.json").read_text())
        assert report["overall"] == "FAIL"
        assert report["layers"]["L0_smoke"] is False
        assert report["layers"]["L1_coverage"] is False

    def test_unit_crash_fails_l1_unit(self, tmp_path) -> None:
        (tmp_path / "smoke.json").write_text(json.dumps({
            "installer_smoke_pass": True,
            "release_smoke_pass": True,
            "wheel_smoke_pass": True,
            "sdist_smoke_pass": True,
            "smoke_duration_s": 1.0,
        }))
        (tmp_path / "unit.json").write_text(
            json.dumps({"passed": 0, "failed": 0, "exit_code": 2})
        )
        (tmp_path / "coverage.json").write_text(json.dumps({"coverage_pct": 85}))
        (tmp_path / "mms.json").write_text(json.dumps([
            {"name": "x", "observed_p": 2.0, "target": 2.0},
        ]))
        (tmp_path / "vv.json").write_text(json.dumps([
            {"case": "y", "validated": True},
        ]))
        (tmp_path / "drift.json").write_text(json.dumps({"drift_score": 0.05}))
        (tmp_path / "security.json").write_text(json.dumps({"security_findings": 0}))

        script = Path(__file__).resolve().parent.parent / "scripts" / "emit_verification_report.py"
        rc = subprocess.run(
            [sys.executable, str(script), str(tmp_path)],
            capture_output=True, text=True, check=False,
        )

        assert rc.returncode == 1
        report = json.loads((tmp_path / "verification_report.json").read_text())
        assert report["overall"] == "FAIL"
        assert report["layers"]["L1_unit"] is False
        assert report["details"]["unit"]["exit_code"] == 2

    def test_coverage_skipped_is_reported_as_na_pass(self, tmp_path) -> None:
        (tmp_path / "smoke.json").write_text(json.dumps({
            "installer_smoke_pass": True,
            "release_smoke_pass": True,
            "wheel_smoke_pass": True,
            "sdist_smoke_pass": True,
            "smoke_duration_s": 1.0,
        }))
        (tmp_path / "unit.json").write_text(
            json.dumps({"passed": 100, "failed": 0, "exit_code": 0})
        )
        (tmp_path / "coverage.json").write_text(
            json.dumps({"coverage_pct": 0, "coverage_skipped": True})
        )
        (tmp_path / "mms.json").write_text(json.dumps([
            {"name": "x", "observed_p": 2.0, "target": 2.0},
        ]))
        (tmp_path / "vv.json").write_text(json.dumps([
            {"case": "y", "validated": True},
        ]))
        (tmp_path / "drift.json").write_text(json.dumps({"drift_score": 0.05}))
        (tmp_path / "security.json").write_text(json.dumps({"security_findings": 0}))

        script = Path(__file__).resolve().parent.parent / "scripts" / "emit_verification_report.py"
        rc = subprocess.run(
            [sys.executable, str(script), str(tmp_path)],
            capture_output=True, text=True, check=False,
        )

        assert rc.returncode == 0
        report = json.loads((tmp_path / "verification_report.json").read_text())
        assert report["overall"] == "PASS"
        assert report["layers"]["L1_coverage"] is True
        assert report["details"]["coverage_skipped"] is True

    def test_coverage_command_failure_fails_l1_coverage(self, tmp_path) -> None:
        (tmp_path / "smoke.json").write_text(json.dumps({
            "installer_smoke_pass": True,
            "release_smoke_pass": True,
            "wheel_smoke_pass": True,
            "sdist_smoke_pass": True,
            "smoke_duration_s": 1.0,
        }))
        (tmp_path / "unit.json").write_text(
            json.dumps({"passed": 100, "failed": 0, "exit_code": 0})
        )
        (tmp_path / "coverage.json").write_text(
            json.dumps(
                {
                    "coverage_pct": 100,
                    "coverage_skipped": False,
                    "coverage_exit_code": 2,
                }
            )
        )
        (tmp_path / "mms.json").write_text(json.dumps([
            {"name": "x", "observed_p": 2.0, "target": 2.0},
        ]))
        (tmp_path / "vv.json").write_text(json.dumps([
            {"case": "y", "validated": True},
        ]))
        (tmp_path / "drift.json").write_text(json.dumps({"drift_score": 0.05}))
        (tmp_path / "security.json").write_text(json.dumps({"security_findings": 0}))

        script = Path(__file__).resolve().parent.parent / "scripts" / "emit_verification_report.py"
        rc = subprocess.run(
            [sys.executable, str(script), str(tmp_path)],
            capture_output=True, text=True, check=False,
        )

        assert rc.returncode == 1
        report = json.loads((tmp_path / "verification_report.json").read_text())
        assert report["overall"] == "FAIL"
        assert report["layers"]["L1_coverage"] is False
        assert report["details"]["coverage_exit_code"] == 2
