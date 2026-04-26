"""Compose 5-layer verification report from raw layer outputs.

Reads JSON inputs from `verify_artifacts/` (one file per layer) and writes
`verification_report.{json,md}` consumable by CI as an artifact.

Layer input files (any missing layer is skipped, not failed):
- unit.json:        {"passed": int, "failed": int}
- coverage.json:    {"coverage_pct": int}
- mms.json:         [{"name": str, "observed_p": float, "target": float}, ...]
- vv.json:          [{"case": str, "validated": bool}, ...]
- drift.json:       {"drift_score": float}
- security.json:    {"security_findings": int}

Examples:
    python3 scripts/emit_verification_report.py verify_artifacts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from naviertwin.utils.verification_report import build_report, to_markdown  # noqa: E402


def _load(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def main(root: str = "verify_artifacts") -> int:
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)

    unit = _load(p / "unit.json", {"passed": 0, "failed": 0})
    coverage = _load(p / "coverage.json", {"coverage_pct": 0})
    mms = _load(p / "mms.json", [])
    vv = _load(p / "vv.json", [])
    drift = _load(p / "drift.json", {"drift_score": 0.0})
    security = _load(p / "security.json", {"security_findings": 0})

    report = build_report(
        unit=unit,
        coverage_pct=int(coverage.get("coverage_pct", 0)),
        mms_results=mms,
        vv_results=vv,
        drift_score=float(drift.get("drift_score", 0.0)),
        security_findings=int(security.get("security_findings", 0)),
    )

    (p / "verification_report.json").write_text(json.dumps(report, indent=2))
    (p / "verification_report.md").write_text(to_markdown(report))
    print(f"verdict: {report['overall']}")
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "verify_artifacts"
    sys.exit(main(root))
