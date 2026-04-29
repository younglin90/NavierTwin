"""GUI tests for the customer onboarding tutorial entry point."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


class _DummyWizard:
    def __init__(self, result: int) -> None:
        self.result = result
        self.calls = 0

    def exec(self) -> int:
        self.calls += 1
        return self.result


def test_help_menu_exposes_tutorial_action(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    actions = [
        action.text()
        for action in win._help_menu.actions()
        if not action.isSeparator()
    ]
    assert any("튜토리얼" in text for text in actions)


def test_tutorial_action_executes_wizard(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    wizard = _DummyWizard(result=1)

    monkeypatch.setattr(win, "_create_tutorial_wizard", lambda: wizard)

    win._show_tutorial()

    assert wizard.calls == 1
    assert win._status_label.text() == "튜토리얼 완료"


def test_tutorial_cancel_updates_status(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    wizard = _DummyWizard(result=0)

    monkeypatch.setattr(win, "_create_tutorial_wizard", lambda: wizard)

    win._show_tutorial()

    assert wizard.calls == 1
    assert win._status_label.text() == "튜토리얼 닫힘"
