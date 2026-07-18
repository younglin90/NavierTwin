"""R659 — History dialog + replay + preset save 통합 검증."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


class TestHistoryDialog:
    def test_dialog_creation(self, qtbot) -> None:
        from naviertwin.gui.widgets.history_dialog import HistoryDialog

        entries = [
            {"timestamp": "2026-01-01T00:00:00", "op": "psd_welch",
             "status": "ok", "kwargs_summary": {"fs": 100.0},
             "result_summary": {"frequency": "[5]"}},
            {"timestamp": "2026-01-01T00:01:00", "op": "denoise",
             "status": "error", "error": "boom",
             "kwargs_summary": {"window_length": 11}},
        ]
        dlg = HistoryDialog(entries)
        qtbot.addWidget(dlg)
        assert dlg._table.rowCount() == 2

    def test_empty_entries(self, qtbot) -> None:
        from naviertwin.gui.widgets.history_dialog import HistoryDialog

        dlg = HistoryDialog([])
        qtbot.addWidget(dlg)
        assert dlg._table.rowCount() == 0

    def test_replay_signal(self, qtbot) -> None:
        from naviertwin.gui.widgets.history_dialog import HistoryDialog

        entries = [
            {"timestamp": "2026-01-01T00:00:00", "op": "psd_welch",
             "status": "ok", "kwargs_summary": {"fs": 100.0}},
        ]
        dlg = HistoryDialog(entries)
        qtbot.addWidget(dlg)
        # Select first row
        dlg._table.selectRow(0)
        emitted: list = []
        dlg.replay_requested.connect(emitted.append)
        dlg._on_replay_clicked()
        assert emitted == [0]

    def test_selection_updates_detail(self, qtbot) -> None:
        from naviertwin.gui.widgets.history_dialog import HistoryDialog

        entries = [
            {"timestamp": "T1", "op": "psd_welch", "status": "ok",
             "kwargs_summary": {"fs": 100.0},
             "result_summary": {"frequency": "[5]"}},
        ]
        dlg = HistoryDialog(entries)
        qtbot.addWidget(dlg)
        dlg._table.selectRow(0)
        text = dlg._detail.toPlainText()
        assert "psd_welch" in text
        assert "100" in text


class TestPanelHistoryButton:
    def test_show_history_opens_dialog(self, qtbot, monkeypatch) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # 한 번 실행 → 이력 1개
        panel._op_list.setCurrentRow(0)
        panel._on_run_clicked()

        # exec()을 mock해서 즉시 닫힘
        monkeypatch.setattr(
            "naviertwin.gui.widgets.history_dialog.HistoryDialog.exec",
            lambda self: 0,
        )
        panel._on_show_history()  # 예외 안 나면 통과

    def test_replay_restores_op_and_params(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # eof 실행 with n_modes=7
        items = panel._op_list.findItems(
            "eof",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items[0])
        panel._param_widgets["n_modes"].setValue(7)
        panel._on_run_clicked()
        # 다른 op로 이동
        panel._op_list.setCurrentRow(0)
        # 이력에서 재실행
        panel._replay_from_history(0)
        # eof로 돌아왔고 n_modes=7
        assert panel._op_list.currentItem().text() == "eof"
        assert panel._param_widgets["n_modes"].value() == 7

    def test_replay_invalid_idx_noop(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # 이력 비어있음 → 인덱스 0도 noop
        panel._replay_from_history(0)
        panel._replay_from_history(-1)


class TestSavePresetButton:
    def test_save_preset_with_input(self, qtbot, monkeypatch) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        items = panel._op_list.findItems(
            "denoise",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items[0])
        panel._param_widgets["window_length"].setValue(15)
        # QInputDialog mock
        monkeypatch.setattr(
            "PySide6.QtWidgets.QInputDialog.getText",
            lambda *a, **kw: ("my_smooth", True),
        )
        panel._on_save_preset()
        # 콤보에 추가됨
        names = [
            panel._preset_combo.itemText(i)
            for i in range(panel._preset_combo.count())
        ]
        assert "my_smooth" in names

    def test_save_preset_cancel(self, qtbot, monkeypatch) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._op_list.setCurrentRow(0)
        # 사용자가 취소
        monkeypatch.setattr(
            "PySide6.QtWidgets.QInputDialog.getText",
            lambda *a, **kw: ("", False),
        )
        panel._on_save_preset()  # 예외 안 나면 통과

    def test_save_preset_no_scalar_params(self, qtbot, monkeypatch) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        items = panel._op_list.findItems(
            "gof_normality",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items[0])
        # scalar param 없는 op → 메시지만
        panel._on_save_preset()
        text = panel._result_text.toPlainText()
        assert "scalar 파라미터 없음" in text


class TestRegression:
    def test_existing_smoke_passes(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        for i in range(panel._op_list.count()):
            panel._op_list.setCurrentRow(i)
            panel._on_run_clicked()
            txt = panel._result_text.toPlainText()
            assert "실행 실패" not in txt
