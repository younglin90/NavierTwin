"""GUI tests for Help-menu update checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")


def _metadata(
    version: str = "4.2.59",
    channel: str = "stable",
    url: str = "https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/NavierTwinSetup.exe",
) -> dict[str, object]:
    return {
        "version": version,
        "channel": channel,
        "url": url,
        "sha256": "a" * 64,
        "notes": "Example metadata for offline update-check smoke validation.",
        "signature": {
            "algorithm": "ed25519",
            "key_id": "naviertwin-release-2026q2",
            "value": (
                "r96mhyh2zIQDcaSd/9b1ExTOngQRiNM7ugU5wFZmckVda8suThYw7TJ8Xtp4kGcWNAaY6n8JhRG2yoHdMsW8BQ=="
            ),
        },
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
    assert "다운로드 열기" in message
    assert "설치파일 검증" in message


def test_main_window_update_check_path_surfaces_result(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from naviertwin.gui.main_window import MainWindow

    path = tmp_path / "release.json"
    path.write_text(json.dumps(_metadata()), encoding="utf-8")
    messages: list[tuple[str, str, object]] = []

    def capture_update_dialog(
        self: object,
        title: str,
        text: str,
        result: object,
    ) -> None:
        messages.append((title, text, result))

    monkeypatch.setattr(
        MainWindow,
        "_show_update_available_dialog",
        capture_update_dialog,
    )
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._check_for_updates_path(path)

    assert messages
    assert messages[0][0] == "업데이트 사용 가능"
    assert "4.2.59" in messages[0][1]
    assert getattr(messages[0][2], "url").endswith("/NavierTwinSetup.exe")
    assert "업데이트 사용 가능" in win._status_label.text()


def test_main_window_update_handoff_opens_download_url(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PySide6.QtGui import QDesktopServices

    from naviertwin.gui.main_window import MainWindow

    url = "https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/NavierTwinSetup.exe"
    opened: list[str] = []

    def capture_open(qurl: object) -> bool:
        opened.append(qurl.toString())
        return True

    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(capture_open))
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    assert win._open_update_download(url) is True
    assert opened == [url]
    assert "열었습니다" in win._status_label.text()


def test_main_window_update_handoff_copies_download_url(qtbot) -> None:
    from PySide6.QtWidgets import QApplication

    from naviertwin.gui.main_window import MainWindow

    url = "https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/NavierTwinSetup.exe"
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    QApplication.clipboard().clear()

    assert win._copy_update_url(url) is True
    assert QApplication.clipboard().text() == url
    assert "복사했습니다" in win._status_label.text()


def test_main_window_update_artifact_verification_succeeds(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PySide6.QtWidgets import QMessageBox

    import naviertwin.utils.updater as updater
    from naviertwin.gui.main_window import MainWindow
    from naviertwin.utils.updater import UpdateCheckResult

    data = b"downloaded setup bytes"
    artifact = tmp_path / "NavierTwinSetup.exe"
    artifact.write_bytes(data)
    messages: list[tuple[str, str]] = []

    def capture_information(parent: object, title: str, text: str) -> None:
        messages.append((title, text))

    monkeypatch.setattr(QMessageBox, "information", capture_information)
    monkeypatch.setattr(updater.platform, "system", lambda: "Linux")
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    result = UpdateCheckResult(
        current_version="4.2.58",
        latest_version="4.2.59",
        channel="stable",
        update_available=True,
        url="https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/NavierTwinSetup.exe",
        sha256=hashlib.sha256(data).hexdigest(),
        installer_signing={
            "publisher": "NavierTwin Contributors",
            "certificate_thumbprint": "e3" * 20,
            "authenticode_required": True,
        },
    )

    assert win._verify_update_artifact_path(result, artifact) is True
    assert messages
    assert messages[0][0] == "설치파일 검증 성공"
    assert "SHA256" in messages[0][1]
    assert "Authenticode: unavailable" in messages[0][1]
    assert win._status_label.text() == "업데이트 설치파일 검증 성공"


def test_main_window_update_artifact_verification_rejects_mismatch(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow
    from naviertwin.utils.updater import UpdateCheckResult

    artifact = tmp_path / "NavierTwinSetup.exe"
    artifact.write_bytes(b"tampered setup bytes")
    warnings: list[tuple[str, str]] = []

    def capture_warning(parent: object, title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(QMessageBox, "warning", capture_warning)
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    result = UpdateCheckResult(
        current_version="4.2.58",
        latest_version="4.2.59",
        channel="stable",
        update_available=True,
        url="https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/NavierTwinSetup.exe",
        sha256="f" * 64,
    )

    assert win._verify_update_artifact_path(result, artifact) is False
    assert warnings
    assert warnings[0][0] == "설치파일 검증 실패"
    assert "기대값" in warnings[0][1]
    assert win._status_label.text() == "업데이트 설치파일 검증 실패"


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
