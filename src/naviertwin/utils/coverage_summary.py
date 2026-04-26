"""Parse pytest-cov terminal output → per-module coverage table.

Examples:
    >>> from naviertwin.utils.coverage_summary import parse_coverage_text
    >>> txt = '''Name              Stmts   Miss  Cover\\nsrc/foo.py          10      2    80%\\nTOTAL              100     20    80%'''
    >>> rows = parse_coverage_text(txt)
    >>> rows[0]['name']
    'src/foo.py'
"""

from __future__ import annotations

import re

_LINE = re.compile(
    r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)%\s*$"
)


def parse_coverage_text(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        rows.append({
            "name": m.group(1),
            "stmts": int(m.group(2)),
            "miss": int(m.group(3)),
            "cover": int(m.group(4)),
        })
    return rows


def below_threshold(rows: list[dict], *, min_pct: int = 70) -> list[dict]:
    return [r for r in rows if r["name"] != "TOTAL" and r["cover"] < min_pct]


__all__ = ["below_threshold", "parse_coverage_text"]
