"""R648 — MainWindow Post-Tools 탭 통합 검증."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


class TestMainWindowIntegration:
    def test_postproc_tab_present(self, qtbot) -> None:
        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)
        # Post-Tools는 Analyze 내부 하위 탭으로 제공됨
        assert win._postproc_panel is not None
        tab_texts = [
            win._tabs.tabText(i) for i in range(win._tabs.count())
        ]
        assert not any("Post-Tools" in t or "후처리" in t for t in tab_texts)
        assert win._analyze_workbench.indexOf(win._postproc_panel) >= 0

    def test_switch_to_postproc_subtab(self, qtbot) -> None:
        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)
        win._switch_to_analyze_workbench(win._postproc_panel)
        assert win._tabs.currentWidget() is win._analyze_workbench
        assert win._analyze_workbench.currentWidget() is win._postproc_panel

    def test_view_menu_exposes_every_tab(self, qtbot) -> None:
        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)

        view_menu = win._view_menu
        actions = [
            action for action in view_menu.actions()
            if isinstance(action.data(), int)
        ]
        assert len(actions) == win._tabs.count()
        assert actions[-1].data() == win._tabs.count() - 1
        assert "Export" in actions[-1].text() or "내보내기" in actions[-1].text()

    def test_library_route_switches_to_postproc_subtab(self, qtbot) -> None:
        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)

        win._on_library_navigate("Post-Tools")

        assert win._tabs.currentWidget() is win._analyze_workbench
        assert win._analyze_workbench.currentWidget() is win._postproc_panel

    def test_dataset_loaded_connects_postproc_panel(
        self, qtbot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from naviertwin.gui.main_window import MainWindow

        dataset = _FakeDataset()
        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)
        captured: list[object] = []

        monkeypatch.setattr(win._analyze_panel, "set_dataset", lambda _: None)
        monkeypatch.setattr(win._reduce_panel, "set_dataset", lambda _: None)
        monkeypatch.setattr(win._model_panel, "set_dataset", lambda _: None)
        monkeypatch.setattr(win._export_panel, "set_dataset", lambda _: None)
        monkeypatch.setattr(
            win._postproc_panel,
            "set_dataset",
            lambda value: captured.append(value),
        )

        win._on_dataset_loaded(dataset)

        assert captured == [dataset]


class _FakeDataset:
    n_points = 4
    n_cells = 2
    n_time_steps = 1
