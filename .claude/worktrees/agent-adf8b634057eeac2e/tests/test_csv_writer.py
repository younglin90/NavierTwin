"""Round 137 — CSV report writer."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestCSV:
    def test_roundtrip(self, tmp_path: Path) -> None:
        from naviertwin.core.report.csv_writer import read_csv, write_csv

        rows = [
            {"name": "A", "value": 1.5},
            {"name": "B", "value": 2.5},
        ]
        p = tmp_path / "x.csv"
        write_csv(p, rows)
        back = read_csv(p)
        assert len(back) == 2
        assert back[0]["name"] == "A"
        assert float(back[0]["value"]) == 1.5

    def test_metrics_table(self, tmp_path: Path) -> None:
        from naviertwin.core.report.csv_writer import read_csv, write_metrics_table

        metrics = {
            "pod_rbf": {"rmse": 0.1, "r2": 0.95},
            "pod_krig": {"rmse": 0.08, "r2": 0.97},
        }
        p = write_metrics_table(tmp_path / "m.csv", metrics)
        rows = read_csv(p)
        assert len(rows) == 2
        assert "config" in rows[0]

    def test_empty_raises(self, tmp_path: Path) -> None:
        from naviertwin.core.report.csv_writer import write_csv

        with pytest.raises(ValueError):
            write_csv(tmp_path / "e.csv", [])
