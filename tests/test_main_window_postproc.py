"""R648 — MainWindow Post-Tools 탭 통합 검증."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


class TestMainWindowIntegration:
    def test_postproc_tab_present(self, qtbot) -> None:
        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)
        # Post-Tools 탭이 추가됨
        assert win._postproc_panel is not None
        # 탭 텍스트에 "Post-Tools" 포함
        tab_texts = [
            win._tabs.tabText(i) for i in range(win._tabs.count())
        ]
        assert any("Post-Tools" in t for t in tab_texts)

    def test_switch_to_postproc_tab(self, qtbot) -> None:
        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)
        # 마지막 탭으로 전환
        last_idx = win._tabs.count() - 1
        win._tabs.setCurrentIndex(last_idx)
        assert win._tabs.currentWidget() is win._postproc_panel

    def test_view_menu_exposes_every_tab(self, qtbot) -> None:
        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)

        view_menu = win._view_menu
        actions = [action for action in view_menu.actions() if action.data() is not None]
        assert len(actions) == win._tabs.count()
        assert actions[-1].data() == win._tabs.count() - 1
        assert "Post-Tools" in actions[-1].text()

    def test_view_menu_switches_to_postproc_tab(self, qtbot) -> None:
        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)

        last_action = [
            action for action in win._view_menu.actions() if action.data() == win._tabs.count() - 1
        ][0]
        last_action.trigger()
        assert win._tabs.currentWidget() is win._postproc_panel
