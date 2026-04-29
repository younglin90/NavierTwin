"""GUI tests for MainWindow explainability tab wiring."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


class _FakeDataset:
    n_points = 4
    n_cells = 2
    n_time_steps = 1


class _DummySurrogate:
    training_metadata: dict[str, object] = {
        "explainability": {
            "background": [[0.0, 0.0], [1.0, 0.0]],
            "feature_names": ["p0", "p1"],
            "output_index": 0,
        }
    }


def test_main_window_explainability_tab_present(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    assert win._explain_panel is not None
    tab_texts = [win._tabs.tabText(i) for i in range(win._tabs.count())]
    assert any("Explain" in text for text in tab_texts)
    assert win._tabs.widget(win._tabs.count() - 1) is win._postproc_panel


def test_view_menu_switches_to_explainability_tab(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    explain_idx = win._tabs.indexOf(win._explain_panel)
    action = [
        item for item in win._view_menu.actions() if item.data() == explain_idx
    ][0]

    action.trigger()

    assert win._tabs.currentWidget() is win._explain_panel


def test_model_trained_forwards_surrogate_to_explainability(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    surrogate = _DummySurrogate()
    captured: list[object] = []

    monkeypatch.setattr(win._explain_panel, "set_model", captured.append)

    win._on_model_trained("rbf", surrogate)

    assert captured == [surrogate]


def test_dataset_loaded_forwards_dataset_to_explainability(
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
    monkeypatch.setattr(win._postproc_panel, "set_dataset", lambda _: None)
    monkeypatch.setattr(win._explain_panel, "set_dataset", captured.append)

    win._on_dataset_loaded(dataset)

    assert captured == [dataset]
