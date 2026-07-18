"""실험 로그 — run metrics 를 CSV 에 append.

동시 실행 안전 (O_APPEND) 하게 한 줄씩 기록.

Examples:
    >>> from naviertwin.utils.experiment_log import ExperimentLog
    >>> # log = ExperimentLog("exp.csv")
    >>> # log.log(config={"lr": 1e-3}, metrics={"rmse": 0.1})
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentLog:
    """append-only 실험 로그."""

    def __init__(
        self, path: str | Path, *, run_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    def log(
        self,
        *,
        config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        row = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": json.dumps(config or {}, ensure_ascii=False, default=str),
            "metrics": json.dumps(metrics or {}, ensure_ascii=False, default=str),
            "notes": notes,
        }
        file_exists = self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists or self.path.stat().st_size == 0:
                w.writeheader()
            w.writerow(row)
        return row

    def read_all(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def clear(self) -> None:
        if self.path.exists():
            os.remove(self.path)


__all__ = ["ExperimentLog"]
