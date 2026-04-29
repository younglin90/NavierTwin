"""GUI tests for Help-menu update checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")


def _metadata(
    version: str = "4.2.59",
    channel: str = "stable",
    url: str = "https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/NavierTwinSetup.exe",
) -> dict[str, str]:
    return {
        "version": version,
        "channel": channel,
        "url": url,
        "sha256": "b" * 64,
        "notes": "gui smoke",
    }


def test_help_menu_exposes_update_check_action(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    actions = [
        action.text()
        for action in win._help_menu.actions()
        if not action.isSeparator()
    ]
    assert any("업데이트 확인" in text for text in actions)


def test_format_update_check_message_for_available_update() -> None:
    from naviertwin.gui.main_window import format_update_check_message
    from naviertwin.utils.updater import UpdateCheckResult

    title, message = format_update_check_message(
        UpdateCheckResult(
            current_version="4.2.58",
            latest_version="4.2.59",
            channel="stable",
            update_available=True,
            url="https://example.com/NavierTwinSetup.exe",
            sha256="b" * 64,
        )
    )

    assert title == "업데이트 사용 가능"
    assert "4.2.59" in message
    assert "SHA256" in message


def test_main_window_update_check_path_surfaces_result(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    path = tmp_path / "release.json"
    path.write_text(json.dumps(_metadata()), encoding="utf-8")
    messages: list[tuple[str, str]] = []

    def capture_information(parent: object, title: str, text: str) -> None:
        messages.append((title, text))

    monkeypatch.setattr(QMessageBox, "information", capture_information)
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._check_for_updates_path(path)

    assert messages
    assert messages[0][0] == "업데이트 사용 가능"
    assert "4.2.59" in messages[0][1]
    assert "업데이트 사용 가능" in win._status_label.text()


def test_main_window_update_check_path_surfaces_metadata_errors(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    path = tmp_path / "release.json"
    payload = _metadata()
    payload["sha256"] = "invalid"
    path.write_text(json.dumps(payload), encoding="utf-8")
    warnings: list[tuple[str, str]] = []

    def capture_warning(parent: object, title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(QMessageBox, "warning", capture_warning)
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._check_for_updates_path(path)

    assert warnings
    assert warnings[0][0] == "업데이트 확인 실패"
    assert "sha256" in warnings[0][1]
    assert win._status_label.text() == "업데이트 확인 실패"
