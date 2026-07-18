"""R657 — Panel history + chart save 통합 검증."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")


class TestHistoryIntegration:
    def test_history_records_success(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._op_list.setCurrentRow(0)
        panel._on_run_clicked()
        hist = panel.history()
        assert len(hist) == 1
        assert hist[0]["status"] == "ok"

    def test_history_records_failure(self, qtbot, monkeypatch) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._op_list.setCurrentRow(0)
        # facade.run을 항상 실패하도록 monkey patch
        monkeypatch.setattr(
            panel._facade, "run",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated")),
        )
        panel._on_run_clicked()
        hist = panel.history()
        assert any(e["status"] == "error" for e in hist)
        assert any("simulated" in (e.get("error") or "") for e in hist)

    def test_history_max_50(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # 60번 실행 → 50개만 남음
        for _ in range(60):
            panel._op_list.setCurrentRow(0)
            panel._on_run_clicked()
        assert len(panel.history()) == 50


class TestChartSaveButton:
    def test_save_chart_disabled_initially(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        assert not panel._save_chart_btn.isEnabled()

    def test_save_chart_enabled_after_run(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._op_list.setCurrentRow(0)
        panel._on_run_clicked()
        if panel._chart is not None:
            assert panel._save_chart_btn.isEnabled()

    def test_save_chart_writes_file(
        self, qtbot, tmp_path, monkeypatch,
    ) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        items = panel._op_list.findItems(
            "psd_welch",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items[0])
        panel._on_run_clicked()

        out = tmp_path / "chart.png"
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            lambda *a, **kw: (str(out), "PNG (*.png)"),
        )
        panel._on_save_chart()
        assert out.exists()
        assert out.stat().st_size > 100

    def test_save_chart_cancel(self, qtbot, monkeypatch) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._op_list.setCurrentRow(0)
        panel._on_run_clicked()
        # 사용자 취소
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            lambda *a, **kw: ("", ""),
        )
        # 예외 안 던짐
        panel._on_save_chart()


class TestRegression:
    def test_existing_smoke_still_passes(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # 모든 op smoke 회귀
        for i in range(panel._op_list.count()):
            panel._op_list.setCurrentRow(i)
            panel._on_run_clicked()
            txt = panel._result_text.toPlainText()
            assert "실행 실패" not in txt
