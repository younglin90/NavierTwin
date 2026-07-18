"""Run pytest-cov, capture overall + per-module coverage, write JSON baseline.

Output:
    coverage_baseline.json — {"total": int, "modules": [{"name": str, "cover": int}]}
    coverage_weak.json     — modules below threshold (default 50%)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_LINE = re.compile(r"^(\S+\.py)\s+\d+\s+\d+\s+(\d+)%\s*(?:\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)?$")
_TOTAL = re.compile(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%\s*$")


def parse(text: str) -> tuple[int, list[dict]]:
    total = 0
    modules: list[dict] = []
    for line in text.splitlines():
        s = line.strip()
        m = _TOTAL.match(s)
        if m:
            total = int(m.group(1))
            continue
        m = _LINE.match(s)
        if m:
            modules.append({"name": m.group(1), "cover": int(m.group(2))})
    return total, modules


def main(out_root: str = ".", threshold: int = 50) -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-m", "not optional",
         "--cov=src/naviertwin", "--cov-report=term", "--no-header"],
        capture_output=True, text=True, check=False,
    )
    text = proc.stdout + proc.stderr
    total, modules = parse(text)
    weak = [m for m in modules if m["cover"] < threshold]
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage_baseline.json").write_text(
        json.dumps({"total": total, "modules": modules}, indent=2),
    )
    (out_dir / "coverage_weak.json").write_text(
        json.dumps({"threshold": threshold, "weak": weak}, indent=2),
    )
    print(f"total: {total}%, weak (<{threshold}%): {len(weak)} modules")
    return 0 if total >= 70 else 1


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.exit(main(root))
