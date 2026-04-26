"""Aggregate verification report — combine results from all 5 layers.

Examples:
    >>> from naviertwin.utils.verification_report import build_report
    >>> r = build_report(
    ...     unit={'passed': 100, 'failed': 0},
    ...     coverage_pct=85,
    ...     mms_results=[{'name': 'multigrid', 'observed_p': 2.0, 'target': 2.0}],
    ...     vv_results=[{'case': 'cavity', 'validated': True}],
    ...     drift_score=0.05,
    ...     security_findings=0,
    ... )
    >>> r['overall']
    'PASS'
"""

from __future__ import annotations

from typing import Any


def build_report(
    *, unit: dict, coverage_pct: int,
    mms_results: list[dict], vv_results: list[dict],
    drift_score: float, security_findings: int,
) -> dict[str, Any]:
    """Aggregate verdict from per-layer inputs."""
    layers = {}
    layers["L1_unit"] = unit["failed"] == 0
    layers["L1_coverage"] = coverage_pct >= 85
    layers["L2_mms"] = all(
        abs(r["observed_p"] - r["target"]) <= 0.5 for r in mms_results
    )
    layers["L3_vv"] = all(r["validated"] for r in vv_results)
    layers["L4_drift"] = drift_score < 0.2
    layers["L5_security"] = security_findings == 0
    overall = "PASS" if all(layers.values()) else "FAIL"
    return {
        "overall": overall,
        "layers": layers,
        "details": {
            "unit": unit,
            "coverage_pct": coverage_pct,
            "mms_results": mms_results,
            "vv_results": vv_results,
            "drift_score": drift_score,
            "security_findings": security_findings,
        },
    }


def to_markdown(report: dict) -> str:
    lines = ["# Verification Report", "", f"**Overall: {report['overall']}**", "",
             "| Layer | Status |", "| --- | --- |"]
    for k, v in report["layers"].items():
        lines.append(f"| {k} | {'✅' if v else '❌'} |")
    return "\n".join(lines) + "\n"


__all__ = ["build_report", "to_markdown"]
