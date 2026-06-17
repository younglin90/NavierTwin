"""간단한 CSV 리포트 작성기 — pandas 없이 동작.

Examples:
    >>> from pathlib import Path
    >>> from naviertwin.core.report.csv_writer import write_csv
    >>> # write_csv("out.csv", [{"a":1,"b":2},{"a":3,"b":4}])
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, object]],
    *,
    fieldnames: list[str] | None = None,
    encoding: str = "utf-8",
) -> Path:
    """리스트of dict → CSV."""
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("rows 비어있음")
    if fieldnames is None:
        fieldnames = list(rows_list[0].keys())
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding=encoding) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        tuple(
            map(
                lambda row: w.writerow(dict(map(lambda k: (k, row.get(k, "")), fieldnames))),
                rows_list,
            )
        )
    return p


def read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_metrics_table(
    path: str | Path,
    metrics: Mapping[str, Mapping[str, float]],
) -> Path:
    """{'config_name': {'rmse': 0.1, 'r2': 0.9}} → CSV."""
    rows = list(map(lambda item: {"config": item[0], **item[1]}, metrics.items()))
    return write_csv(path, rows)


__all__ = ["write_csv", "read_csv", "write_metrics_table"]
