"""Round 140 — 실험 로거."""

from __future__ import annotations

import json
from pathlib import Path


class TestExperimentLog:
    def test_append_and_read(self, tmp_path: Path) -> None:
        from naviertwin.utils.experiment_log import ExperimentLog

        log = ExperimentLog(tmp_path / "exp.csv", run_id="R001")
        log.log(config={"lr": 1e-3}, metrics={"rmse": 0.1}, notes="first")
        log.log(config={"lr": 5e-4}, metrics={"rmse": 0.08})
        rows = log.read_all()
        assert len(rows) == 2
        assert rows[0]["run_id"] == "R001"
        assert json.loads(rows[0]["metrics"])["rmse"] == 0.1

    def test_clear(self, tmp_path: Path) -> None:
        from naviertwin.utils.experiment_log import ExperimentLog

        log = ExperimentLog(tmp_path / "x.csv")
        log.log(metrics={"a": 1})
        assert log.path.exists()
        log.clear()
        assert not log.path.exists()
