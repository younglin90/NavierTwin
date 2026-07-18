"""GUI tests for customer diagnostics in the Help menu."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_help_menu_exposes_doctor_action(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    actions = [
        action.text()
        for action in win._help_menu.actions()
        if not action.isSeparator()
    ]
    assert any("환경 진단" in text for text in actions)


def test_doctor_action_surfaces_diagnostic_report(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    import naviertwin.utils.doctor as doctor
    from naviertwin.gui.main_window import MainWindow

    monkeypatch.setattr(
        doctor,
        "build_doctor_report",
        lambda include_optional=True: {
            "status": "warn",
            "version": "4.2.58",
            "checks": [
                {"name": "python_version", "status": "ok", "details": {}},
                {"name": "cuda", "status": "warn", "details": {}},
            ],
            "warnings": ["cuda"],
            "errors": [],
        },
    )
    messages: list[tuple[str, str]] = []

    def capture_information(parent: object, title: str, text: str) -> None:
        messages.append((title, text))

    monkeypatch.setattr(QMessageBox, "information", capture_information)
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._show_doctor_report()

    assert messages
    assert messages[0][0] == "NavierTwin 환경 진단"
    assert "python_version: ok" in messages[0][1]
    assert "cuda: warn" in messages[0][1]
    assert win._status_label.text() == "환경 진단: warn"
