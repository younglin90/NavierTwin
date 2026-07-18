"""MainWindow integration tests for the Library tab."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_main_window_library_available_as_help_dialog(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    tab_texts = [win._tabs.tabText(i) for i in range(win._tabs.count())]
    assert win._library_panel is not None
    assert not any("Library" in text or "기능" in text for text in tab_texts)
    assert win._library_panel.parentWidget() is win._library_dialog
    assert win._library_dialog.objectName() == "librarySearchDialog"
    assert any(action.text().startswith("기능 검색") for action in win._help_menu.actions())

    win._show_library_search()

    assert win._library_dialog.isVisible()
    assert win._library_dialog.width() >= 980
    assert win._library_dialog.height() >= 680


def test_library_navigation_switches_nested_post_tools(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._on_library_navigate("Post-Tools")

    assert win._tabs.currentWidget() is win._analyze_workbench
    assert win._analyze_workbench.currentWidget() is win._postproc_panel
    assert win._status_label.text() == "기능 탭 이동: Post-Tools"


def test_dataset_loaded_connects_library_panel(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
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
    if win._postproc_panel is not None:
        monkeypatch.setattr(win._postproc_panel, "set_dataset", lambda _: None)
    monkeypatch.setattr(win._library_panel, "set_dataset", lambda value: captured.append(value))

    win._on_dataset_loaded(dataset)

    assert captured == [dataset]


class _FakeDataset:
    n_points = 4
    n_cells = 2
    n_time_steps = 1
