"""Round 522 — experiment registry."""

from __future__ import annotations


class TestExp:
    def test_log_list(self, tmp_path) -> None:
        from naviertwin.utils.workflow.experiment_registry import ExperimentRegistry

        r = ExperimentRegistry(tmp_path / "r.db")
        rid = r.log_run("e1", {"lr": 0.01}, {"acc": 0.9})
        assert rid > 0
        runs = r.list_runs()
        assert len(runs) == 1
        assert runs[0]["params"]["lr"] == 0.01
        assert runs[0]["metrics"]["acc"] == 0.9
