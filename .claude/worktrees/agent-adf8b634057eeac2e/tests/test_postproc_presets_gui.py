"""R658 — Post-Tools 프리셋 콤보 + 사용자 프리셋 통합 검증."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


class TestPresetCombo:
    def test_combo_populated_for_known_op(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        items = panel._op_list.findItems(
            "psd_welch",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items[0])
        # factory presets로 채워짐
        n = panel._preset_combo.count()
        assert n >= 4  # "(없음)" + 3 factory

    def test_combo_only_default_for_unknown_op(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # gof_normality는 factory preset 없음
        items = panel._op_list.findItems(
            "gof_normality",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items[0])
        # "(없음)"만
        assert panel._preset_combo.count() == 1
        assert panel._preset_combo.itemText(0) == "(없음)"

    def test_preset_applies_to_form(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        items = panel._op_list.findItems(
            "psd_welch",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items[0])
        # high_resolution 프리셋 적용
        panel._preset_combo.setCurrentText("high_resolution")
        vals = panel._read_param_values()
        assert vals["fs"] == 1000.0
        assert vals["nperseg"] == 1024

    def test_none_preset_no_change(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        items = panel._op_list.findItems(
            "denoise",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items[0])
        # 기본값 저장
        before = panel._read_param_values()
        # "(없음)" 선택해도 변경 없음
        panel._preset_combo.setCurrentText("(없음)")
        after = panel._read_param_values()
        assert after == before


class TestUserPreset:
    def test_add_user_preset(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        items = panel._op_list.findItems(
            "eof",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items[0])
        panel._param_widgets["n_modes"].setValue(15)
        panel.add_user_preset("eof", "my_run")
        # 콤보에 추가됨
        names = [
            panel._preset_combo.itemText(i)
            for i in range(panel._preset_combo.count())
        ]
        assert "my_run" in names

    def test_user_preset_persists_until_op_change_back(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        items_eof = panel._op_list.findItems(
            "eof",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items_eof[0])
        panel.add_user_preset("eof", "my_setup", {"n_modes": 11})
        # 다른 op 갔다가 돌아옴
        panel._op_list.setCurrentRow(0)
        panel._op_list.setCurrentItem(items_eof[0])
        names = [
            panel._preset_combo.itemText(i)
            for i in range(panel._preset_combo.count())
        ]
        assert "my_setup" in names

    def test_user_preset_overrides_factory(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        items = panel._op_list.findItems(
            "eof",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        panel._op_list.setCurrentItem(items[0])
        # factory "default"는 n_modes=5
        panel.add_user_preset("eof", "default", {"n_modes": 100})
        panel._preset_combo.setCurrentText("default")
        vals = panel._read_param_values()
        assert vals["n_modes"] == 100


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
