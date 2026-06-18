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
    sep_cells: list[str] = []
    col_idx = 0
    while col_idx < len(columns):
        sep_cells.append("---")
        col_idx += 1
    sep = "| " + " | ".join(sep_cells) + " |"
    rows = []
    run_idx = 0
    while run_idx < len(runs):
        r = runs[run_idx]
        cells: list[str] = []
        col_idx = 0
        while col_idx < len(columns):
            cells.append(str(r.get(columns[col_idx], "")))
            col_idx += 1
        rows.append("| " + " | ".join(cells) + " |")
        run_idx += 1
    return "\n".join([header, sep, *rows]) + "\n"


__all__ = ["compare_md"]
