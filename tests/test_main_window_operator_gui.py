"""GUI tests for neural-operator training handoff behavior."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


class _DummyOperator:
    train_losses_ = [1.0, 0.5]


class _DummySurrogate:
    training_metadata = {
        "validation_metrics": {
            "rmse": 0.1,
            "r2": 0.9,
        }
    }


def test_operator_training_does_not_build_twin_engine(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    win._latest_reducer = object()
    operator = _DummyOperator()
    build_calls: list[object] = []
    export_models: list[object] = []

    def fail_if_called(reducer: object, surrogate: object) -> object:
        build_calls.append((reducer, surrogate))
        raise AssertionError("operator models must not build TwinEngine")

    monkeypatch.setattr(win, "_build_engine", fail_if_called)
    monkeypatch.setattr(
        win._export_panel,
        "set_model",
        lambda value: export_models.append(value),
    )

    win._on_model_trained("fno1d", operator)

    assert build_calls == []
    assert win._latest_operator is operator
    assert win._latest_engine is None
    assert export_models == [operator]
    assert win._tabs.currentWidget() is win._model_panel
    assert "TwinEngine 자동 연결 생략" in win._status_label.text()


def test_surrogate_training_still_builds_twin_engine(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    reducer = object()
    surrogate = _DummySurrogate()
    engine = object()
    win._latest_reducer = reducer
    twin_engines: list[object] = []
    export_engines: list[object] = []

    monkeypatch.setattr(win, "_build_engine", lambda *_: engine)
    monkeypatch.setattr(win._twin_panel, "set_engine", lambda value: twin_engines.append(value))
    monkeypatch.setattr(
        win._export_panel,
        "set_engine",
        lambda value: export_engines.append(value),
    )

    win._on_model_trained("rbf", surrogate)

    assert win._latest_surrogate is surrogate
    assert win._latest_engine is engine
    assert twin_engines == [engine]
    assert export_engines == [engine]
    assert "TwinEngine 연결 완료" in win._status_label.text()
