"""Round 17 — MainWindow i18n + 비교 탭 통합 (import-only)."""

from __future__ import annotations

import pytest


class TestMainWindowAssembly:
    """QApplication 필요하므로 import/attribute 수준 검증만."""

    def test_language_switch_method_exists(self) -> None:
        pytest.importorskip("PySide6")
        from naviertwin.gui.main_window import MainWindow

        assert hasattr(MainWindow, "set_language")
        assert hasattr(MainWindow, "update_compare_dashboard")

    def test_translator_inlined(self) -> None:
        from naviertwin.utils.i18n import Translator

        t = Translator(lang="en")
        assert t("panel.import") == "Import"
        assert t("panel.post_tools") == "Post-Tools"
        assert t("view.tab_action") == "{title} Tab"
        t.set_language("ko")
        assert t("panel.import") == "불러오기"
        assert t("panel.post_tools") == "후처리"
        assert t("view.tab_action").endswith("탭")

    def test_startup_applies_language_and_theme_config(
        self, qtbot, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        import naviertwin.gui.main_window as main_window
        from naviertwin.utils.config import NavierTwinConfig, save_config

        config_path = tmp_path / "config.json"
        save_config(NavierTwinConfig(language="en", theme="light"), config_path)
        loaded_themes: list[str] = []

        def capture_stylesheet(theme: str = "dark") -> str:
            loaded_themes.append(theme)
            return f"/* {theme} theme */"

        app = QApplication.instance()
        original_stylesheet = app.styleSheet()  # type: ignore[union-attr]
        monkeypatch.setattr(main_window, "_load_stylesheet", capture_stylesheet)
        try:
            win = main_window.MainWindow(
                confirm_on_close=False,
                config_path=config_path,
            )
            qtbot.addWidget(win)

            assert loaded_themes == ["light"]
            assert win._t.lang == "en"
            assert win.windowTitle() == "NavierTwin — CFD Digital Twin"
            assert win._tabs.tabText(0) == "① Import"
            assert app.styleSheet() == "/* light theme */"  # type: ignore[union-attr]
        finally:
            app.setStyleSheet(original_stylesheet)  # type: ignore[union-attr]

    def test_view_menu_exposes_language_and_theme_preferences(self, qtbot) -> None:
        pytest.importorskip("PySide6")
        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)

        actions = [
            action.text()
            for action in win._view_menu.actions()
            if not action.isSeparator()
        ]
        assert "다크 테마" in actions
        assert "라이트 테마" in actions
        assert "한국어" in actions
        assert "English" in actions

    def test_view_menu_language_and_theme_preferences_persist(
        self, qtbot, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("PySide6")
        from PySide6.QtGui import QAction
        from PySide6.QtWidgets import QApplication

        import naviertwin.gui.main_window as main_window
        from naviertwin.utils.config import NavierTwinConfig, load_config, save_config

        def action_with_data(data: str) -> QAction:
            for action in win._view_menu.actions():
                if action.data() == data:
                    return action
            raise AssertionError(f"missing view-menu action: {data}")

        config_path = tmp_path / "config.json"
        save_config(NavierTwinConfig(language="ko", theme="dark"), config_path)
        loaded_themes: list[str] = []

        def capture_stylesheet(theme: str = "dark") -> str:
            loaded_themes.append(theme)
            return f"/* {theme} theme */"

        app = QApplication.instance()
        original_stylesheet = app.styleSheet()  # type: ignore[union-attr]
        monkeypatch.setattr(main_window, "_load_stylesheet", capture_stylesheet)
        try:
            win = main_window.MainWindow(
                confirm_on_close=False,
                config_path=config_path,
            )
            qtbot.addWidget(win)

            action_with_data("light").trigger()
            assert win._config.theme == "light"
            assert loaded_themes[-1] == "light"
            assert load_config(config_path).theme == "light"

            action_with_data("en").trigger()
            loaded = load_config(config_path)
            assert loaded.language == "en"
            assert loaded.theme == "light"
            assert win._tabs.tabText(0) == "① Import"
            assert win._view_menu.title() == "View(&V)"
            en_actions = [
                action.text()
                for action in win._view_menu.actions()
                if not action.isSeparator()
            ]
            assert "Light Theme" in en_actions
            assert "Korean" in en_actions
            assert win._status_label.text() == "언어 변경: en"
        finally:
            app.setStyleSheet(original_stylesheet)  # type: ignore[union-attr]

    def test_language_switch_updates_optional_workflow_tabs_and_view_menu(self, qtbot) -> None:
        pytest.importorskip("PySide6")
        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)

        workflow_expectations = [
            ("① 불러오기", "① Import"),
            ("② 분석", "② Analyze"),
            ("③ 차원축소", "③ Reduce"),
            ("④ 모델", "④ Model"),
            ("⑤ 디지털트윈", "⑤ Twin"),
            ("⑥ 내보내기", "⑥ Export"),
        ]

        win.set_language("ko")
        ko_titles = [win._tabs.tabText(i) for i in range(win._tabs.count())]
        ko_actions = [
            action.text()
            for action in win._view_menu.actions()
            if not action.isSeparator()
        ]

        for ko_title, _ in workflow_expectations:
            assert ko_title in ko_titles
            assert f"{ko_title} 탭" in ko_actions

        win.set_language("en")
        en_titles = [win._tabs.tabText(i) for i in range(win._tabs.count())]
        en_actions = [
            action.text()
            for action in win._view_menu.actions()
            if not action.isSeparator()
        ]

        for _, en_title in workflow_expectations:
            assert en_title in en_titles
            assert f"{en_title} Tab" in en_actions

    def test_compare_dashboard_is_nested_under_model_tab(self, qtbot) -> None:
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QTabWidget

        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)

        top_titles = [win._tabs.tabText(i) for i in range(win._tabs.count())]
        assert not any("Compare" in title or "비교" in title for title in top_titles)
        assert not any("Explain" in title or "설명" in title for title in top_titles)
        assert win._tabs.indexOf(win._model_workbench) >= 0

        assert isinstance(win._model_workbench, QTabWidget)
        model_subtabs = [
            win._model_workbench.tabText(i)
            for i in range(win._model_workbench.count())
        ]
        assert any("Model" in title or "모델" in title for title in model_subtabs)
        assert any("Compare" in title or "비교" in title for title in model_subtabs)
        assert any("Explain" in title or "설명" in title for title in model_subtabs)

    def test_post_tools_is_nested_under_analyze_tab(self, qtbot) -> None:
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QTabWidget

        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)

        top_titles = [win._tabs.tabText(i) for i in range(win._tabs.count())]
        assert not any("Post-Tools" in title or "후처리" in title for title in top_titles)
        assert win._tabs.indexOf(win._analyze_workbench) >= 0

        assert isinstance(win._analyze_workbench, QTabWidget)
        analyze_subtabs = [
            win._analyze_workbench.tabText(i)
            for i in range(win._analyze_workbench.count())
        ]
        assert any("Analyze" in title or "분석" in title for title in analyze_subtabs)
        assert any("Post-Tools" in title or "후처리" in title for title in analyze_subtabs)

    def test_pipeline_tabs_wrap_to_multiple_rows(self, qtbot) -> None:
        pytest.importorskip("PySide6")
        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)

        assert win._tabs.count() == 6
        assert win._tabs._tab_layout.rowCount() >= 2
        assert all(button.minimumHeight() >= 38 for button in win._tabs._buttons)

    def test_library_is_help_dialog_not_workflow_tab(self, qtbot) -> None:
        pytest.importorskip("PySide6")
        from naviertwin.gui.main_window import MainWindow

        win = MainWindow(confirm_on_close=False)
        qtbot.addWidget(win)

        top_titles = [win._tabs.tabText(i) for i in range(win._tabs.count())]
        assert not any("Library" in title or "기능" in title for title in top_titles)
        assert win._library_panel.parentWidget() is win._library_dialog
        assert not win._library_dialog.isVisible()

        win._show_library_search()

        assert win._library_dialog.isVisible()
        assert win._library_dialog.width() >= 980
        assert win._library_dialog.height() >= 680
