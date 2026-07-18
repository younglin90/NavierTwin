"""Round 530 — AA category milestone: workflow (R521-R529) e2e."""

from __future__ import annotations


class TestMilestoneAA:
    def test_imports(self) -> None:
        from naviertwin.utils.workflow import (  # noqa: F401
            artifact_zip,
            budget,
            checkpoint_mgr,
            compare_report,
            dag_runner,
            experiment_registry,
            notify,
            resume,
            sweep,
        )

    def test_full_run_e2e(self, tmp_path) -> None:
        from naviertwin.utils.workflow.compare_report import compare_md
        from naviertwin.utils.workflow.dag_runner import DAGRunner
        from naviertwin.utils.workflow.experiment_registry import ExperimentRegistry
        from naviertwin.utils.workflow.sweep import grid_sweep

        reg = ExperimentRegistry(tmp_path / "r.db")
        for params in grid_sweep({"lr": [0.01, 0.1]}):
            r = DAGRunner()
            r.add("model", lambda i, p=params: 1 - p["lr"])
            r.add("metric", lambda i: {"acc": i["model"]}, deps=["model"])
            out = r.run()
            reg.log_run("sweep", params, out["metric"])

        runs = reg.list_runs()
        rows = [{"name": str(r["params"]["lr"]),
                  "acc": r["metrics"]["acc"]} for r in runs]
        md = compare_md(rows, columns=["name", "acc"])
        assert "| 0.01 |" in md or "0.01" in md
        assert len(runs) == 2
