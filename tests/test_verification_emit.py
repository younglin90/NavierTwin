"""Round 571 — verification report emitter (CI artifact)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


class TestEmitter:
    def test_pass_path(self, tmp_path) -> None:
        # build inputs
        (tmp_path / "unit.json").write_text(json.dumps({"passed": 100, "failed": 0}))
        (tmp_path / "coverage.json").write_text(json.dumps({"coverage_pct": 78}))
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
        md = (tmp_path / "verification_report.md").read_text()
        assert "PASS" in md

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
        assert report["layers"]["L1_coverage"] is False
