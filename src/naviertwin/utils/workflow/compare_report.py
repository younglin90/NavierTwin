"""Comparison report — markdown table from list of run dicts.

Examples:
    >>> from naviertwin.utils.workflow.compare_report import compare_md
    >>> runs = [{'name': 'A', 'acc': 0.9}, {'name': 'B', 'acc': 0.85}]
    >>> '| name |' in compare_md(runs, columns=['name', 'acc'])
    True
"""

from __future__ import annotations

from collections.abc import Sequence


def compare_md(runs: Sequence[dict], *, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |"
            for r in runs]
    return "\n".join([header, sep, *rows]) + "\n"


__all__ = ["compare_md"]
