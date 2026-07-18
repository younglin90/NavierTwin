"""Experiment registry — SQLite store of runs.

Examples:
    >>> import tempfile, pathlib
    >>> from naviertwin.utils.workflow.experiment_registry import ExperimentRegistry
    >>> with tempfile.TemporaryDirectory() as d:
    ...     reg = ExperimentRegistry(pathlib.Path(d) / 'r.db')
    ...     reg.log_run('exp1', {'lr': 0.01}, {'acc': 0.9})
    ...     len(reg.list_runs())
    1
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class ExperimentRegistry:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS runs ("
                "id INTEGER PRIMARY KEY, name TEXT, params TEXT, metrics TEXT)"
            )

    def log_run(self, name: str, params: dict, metrics: dict) -> int:
        with sqlite3.connect(self.db_path) as c:
            cur = c.execute(
                "INSERT INTO runs(name, params, metrics) VALUES (?, ?, ?)",
                (name, json.dumps(params), json.dumps(metrics)),
            )
            return int(cur.lastrowid)

    def list_runs(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute(
                "SELECT id, name, params, metrics FROM runs",
            ).fetchall()
        out: list[dict] = []
        idx = 0
        while idx < len(rows):
            r = rows[idx]
            out.append({
                "id": r[0],
                "name": r[1],
                "params": json.loads(r[2]),
                "metrics": json.loads(r[3]),
            })
            idx += 1
        return out


__all__ = ["ExperimentRegistry"]
