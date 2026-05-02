"""R657 — RunHistory 검증."""

from __future__ import annotations

import json

import numpy as np
import pytest


class TestRunHistory:
    def test_record_and_len(self) -> None:
        from naviertwin.core.post_process_history import RunHistory

        hist = RunHistory(max_entries=10)
        hist.record("psd_welch", {"fs": 100.0}, {"frequency": np.zeros(5)})
        assert len(hist) == 1

    def test_max_entries_eviction(self) -> None:
        from naviertwin.core.post_process_history import RunHistory

        hist = RunHistory(max_entries=3)
        for i in range(5):
            hist.record(f"op_{i}", {}, {"x": np.array([i])})
        assert len(hist) == 3
        # 마지막 3개만 남음
        ops = [e["op"] for e in hist.entries()]
        assert ops == ["op_2", "op_3", "op_4"]

    def test_invalid_max_entries(self) -> None:
        from naviertwin.core.post_process_history import RunHistory

        with pytest.raises(ValueError, match="max_entries"):
            RunHistory(max_entries=0)

    def test_record_with_error(self) -> None:
        from naviertwin.core.post_process_history import RunHistory

        hist = RunHistory()
        hist.record("bad_op", {}, None, status="error", error="boom")
        assert hist.last()["status"] == "error"
        assert hist.last()["error"] == "boom"
        assert hist.last()["result_summary"] is None

    def test_kwargs_summary_ndarray(self) -> None:
        from naviertwin.core.post_process_history import RunHistory

        hist = RunHistory()
        hist.record("op", {"data": np.zeros((10, 5))}, {"x": np.zeros(3)})
        last = hist.last()
        assert "ndarray" in last["kwargs_summary"]["data"]

    def test_kwargs_summary_callable(self) -> None:
        from naviertwin.core.post_process_history import RunHistory

        hist = RunHistory()
        def my_fn(x):
            return x

        hist.record("op", {"f": my_fn}, {"x": np.zeros(3)})
        last = hist.last()
        assert "callable" in last["kwargs_summary"]["f"]

    def test_result_summary_includes_stats(self) -> None:
        from naviertwin.core.post_process_history import RunHistory

        hist = RunHistory()
        hist.record("op", {}, {"y": np.array([1.0, 2.0, 3.0])})
        summary = hist.last()["result_summary"]["y"]
        assert summary["shape"] == [3]
        assert summary["mean"] == 2.0

    def test_filter_by_op(self) -> None:
        from naviertwin.core.post_process_history import RunHistory

        hist = RunHistory()
        hist.record("a", {}, {"x": np.array([1])})
        hist.record("b", {}, {"x": np.array([1])})
        hist.record("a", {}, {"x": np.array([1])})
        a_only = hist.filter_by_op("a")
        assert len(a_only) == 2

    def test_filter_by_status(self) -> None:
        from naviertwin.core.post_process_history import RunHistory

        hist = RunHistory()
        hist.record("a", {}, {"x": np.array([1])}, status="ok")
        hist.record("b", {}, None, status="error", error="x")
        ok = hist.filter_by_status("ok")
        err = hist.filter_by_status("error")
        assert len(ok) == 1
        assert len(err) == 1

    def test_clear(self) -> None:
        from naviertwin.core.post_process_history import RunHistory

        hist = RunHistory()
        hist.record("a", {}, {"x": np.array([1])})
        hist.clear()
        assert len(hist) == 0
        assert hist.last() is None

    def test_save_and_load(self, tmp_path) -> None:
        from naviertwin.core.post_process_history import RunHistory

        hist = RunHistory()
        hist.record("a", {"fs": 100.0}, {"y": np.array([1.0, 2.0])})
        hist.record("b", {}, None, status="error", error="boom")
        path = tmp_path / "hist.json"
        hist.save_json(path)
        # 유효 JSON
        data = json.loads(path.read_text())
        assert len(data) == 2
        # 복원
        hist2 = RunHistory.load_json(path)
        assert len(hist2) == 2
        assert hist2.entries()[0]["op"] == "a"


class TestChartSaveFigure:
    def test_save_png(self, qtbot, tmp_path) -> None:
        pytest.importorskip("PySide6")
        pytest.importorskip("matplotlib")
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        chart.render("psd_welch", {
            "frequency": np.linspace(0, 10, 50),
            "psd": np.exp(-np.linspace(0, 10, 50)),
        })
        out = tmp_path / "fig.png"
        chart.save_figure(str(out))
        assert out.exists()
        assert out.stat().st_size > 100

    def test_save_svg(self, qtbot, tmp_path) -> None:
        pytest.importorskip("PySide6")
        pytest.importorskip("matplotlib")
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        chart.render("psd_welch", {
            "frequency": np.linspace(0, 10, 50),
            "psd": np.exp(-np.linspace(0, 10, 50)),
        })
        out = tmp_path / "fig.svg"
        chart.save_figure(str(out))
        assert out.exists()
        # SVG는 텍스트 시작
        assert out.read_text().startswith("<?xml") or "<svg" in out.read_text()

    def test_save_no_figure_raises(self, qtbot, tmp_path) -> None:
        pytest.importorskip("PySide6")
        pytest.importorskip("matplotlib")
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        with pytest.raises(ValueError, match="figure"):
            chart.save_figure(str(tmp_path / "x.png"))
