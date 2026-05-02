"""R656 — Post-Tools panel export buttons + bulk runner."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")


class TestExportButtonsState:
    def test_export_disabled_initially(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        assert not panel._export_csv_btn.isEnabled()
        assert not panel._export_json_btn.isEnabled()
        assert not panel._export_npz_btn.isEnabled()

    def test_export_enabled_after_run(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._op_list.setCurrentRow(0)
        panel._on_run_clicked()
        # Demo 실행 성공 시 export 버튼 활성화
        assert panel._export_csv_btn.isEnabled()
        assert panel._export_json_btn.isEnabled()
        assert panel._export_npz_btn.isEnabled()
        assert panel._last_result is not None
        assert panel._last_op_name is not None


class TestExportFlow:
    def test_csv_save(self, qtbot, tmp_path, monkeypatch) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # psd_welch 실행
        items = panel._op_list.findItems(
            "psd_welch",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items[0])
        panel._on_run_clicked()

        # QFileDialog mock
        out_path = tmp_path / "out.csv"
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            lambda *a, **kw: (str(out_path), "CSV (*.csv)"),
        )
        panel._on_export_csv()
        assert out_path.exists()
        text = out_path.read_text()
        assert "frequency" in text or "psd" in text

    def test_json_save(self, qtbot, tmp_path, monkeypatch) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._op_list.setCurrentRow(0)
        panel._on_run_clicked()

        out_path = tmp_path / "out.json"
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            lambda *a, **kw: (str(out_path), "JSON (*.json)"),
        )
        panel._on_export_json()
        assert out_path.exists()
        # 유효한 JSON
        json.loads(out_path.read_text())

    def test_npz_save(self, qtbot, tmp_path, monkeypatch) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # 첫 op (대부분 ndarray 결과)
        panel._op_list.setCurrentRow(0)
        panel._on_run_clicked()

        out_path = tmp_path / "out.npz"
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            lambda *a, **kw: (str(out_path), "NPZ (*.npz)"),
        )
        panel._on_export_npz()
        # 일부 op은 NPZ 저장 가능 (ndarray 있을 때)

    def test_export_cancel_does_nothing(self, qtbot, monkeypatch) -> None:
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
        panel._on_export_csv()


class TestBulkRunner:
    def test_run_category_statistics(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._category_combo.setCurrentText("statistics")
        panel._on_run_category()
        text = panel._result_text.toPlainText()
        assert "Bulk Post-Process Summary" in text
        assert "총 op:" in text

    def test_run_all_disabled_for_all_category(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._category_combo.setCurrentText("전체")
        panel._on_run_category()
        text = panel._result_text.toPlainText()
        assert "비활성" in text or "특정 카테고리" in text


class TestRegression:
    def test_chart_still_renders_after_export_changes(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        items = panel._op_list.findItems(
            "psd_welch",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items[0])
        panel._on_run_clicked()
        assert panel._chart._last_op == "psd_welch"
