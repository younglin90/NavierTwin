"""Round 570 — verification milestone: 5-layer aggregate report e2e."""

from __future__ import annotations


class TestMilestoneVerify:
    def test_imports(self) -> None:
        from naviertwin.core.benchmarks import ghia_cavity  # noqa: F401
        from naviertwin.utils import (  # noqa: F401
            coverage_summary,
            flakiness,
            mutation_smoke,
            repro_manifest,
            supply_audit,
            verification_report,
        )

    def test_pass_report(self) -> None:
        from naviertwin.utils.verification_report import build_report, to_markdown

        r = build_report(
            unit={"passed": 1900, "failed": 0},
            coverage_pct=78,
            mms_results=[
                {"name": "multigrid", "observed_p": 2.0, "target": 2.0},
                {"name": "ssp_rk3", "observed_p": 3.1, "target": 3.0},
            ],
            vv_results=[
                {"case": "cavity_re100", "validated": True},
                {"case": "burgers_shock", "validated": True},
            ],
            drift_score=0.05,
            security_findings=0,
        )
        assert r["overall"] == "PASS"
        md = to_markdown(r)
        assert "# Verification Report" in md
        assert "✅" in md

    def test_fail_report(self) -> None:
        from naviertwin.utils.verification_report import build_report

        r = build_report(
            unit={"passed": 100, "failed": 5},  # failures
            coverage_pct=80,
            mms_results=[{"name": "x", "observed_p": 2.0, "target": 2.0}],
            vv_results=[{"case": "y", "validated": True}],
            drift_score=0.05,
            security_findings=0,
        )
        assert r["overall"] == "FAIL"
        assert r["layers"]["L1_unit"] is False
