"""GUI tests for customer-facing tools menu workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")


def test_tools_menu_exposes_pipeline_demo_and_server_actions(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    actions = [
        action.text()
        for action in win._tools_menu.actions()
        if not action.isSeparator()
    ]
    assert any("파이프라인 데모" in text for text in actions)
    assert any("API 서버 시작" in text for text in actions)
    assert any("API 서버 중지" in text for text in actions)


def test_pipeline_demo_path_runs_cli_and_surfaces_result(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    calls: list[Path] = []
    messages: list[tuple[str, str]] = []

    monkeypatch.setattr(win, "_run_pipeline_demo_cli", lambda outdir: calls.append(outdir) or 0)
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    win._run_pipeline_demo_path(tmp_path)

    assert calls == [tmp_path]
    assert messages
    assert messages[0][0] == "파이프라인 데모 완료"
    assert win._status_label.text() == "파이프라인 데모 완료"


def test_pipeline_demo_path_surfaces_failure(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    warnings: list[tuple[str, str]] = []

    monkeypatch.setattr(win, "_run_pipeline_demo_cli", lambda outdir: 2)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )

    win._run_pipeline_demo_path(tmp_path)

    assert warnings
    assert warnings[0][0] == "파이프라인 데모 실패"
    assert "2" in warnings[0][1]
    assert win._status_label.text() == "파이프라인 데모 실패"


def test_api_server_start_uses_qprocess(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    fake = _FakeProcess(started=True)
    messages: list[tuple[str, str]] = []

    monkeypatch.setattr(win, "_create_api_server_process", lambda: fake)
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    win._start_api_server()

    assert fake.program
    assert fake.args[:3] == ["-m", "naviertwin.main", "server"]
    assert fake.started
    assert messages
    assert win._server_process is fake
    assert "API 서버 실행 중" in win._status_label.text()


def test_api_server_stop_terminates_running_process(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    fake = _FakeProcess(started=True)
    win._server_process = fake

    win._stop_api_server()

    assert fake.terminated
    assert win._server_process is None
    assert win._status_label.text() == "API 서버 중지됨"


class _FakeProcess:
    def __init__(self, *, started: bool) -> None:
        from PySide6.QtCore import QProcess

        self.program = ""
        self.args: list[str] = []
        self.started = False
        self.terminated = False
        self.killed = False
        self._state = (
            QProcess.ProcessState.Running
            if started
            else QProcess.ProcessState.NotRunning
        )

    def state(self):
        return self._state

    def setProgram(self, program: str) -> None:
        self.program = program

    def setArguments(self, args: list[str]) -> None:
        self.args = args

    def start(self) -> None:
        self.started = True

    def waitForStarted(self, timeout: int) -> bool:
        return self._state.name == "Running"

    def terminate(self) -> None:
        from PySide6.QtCore import QProcess

        self.terminated = True
        self._state = QProcess.ProcessState.NotRunning

    def waitForFinished(self, timeout: int) -> bool:
        return True

    def kill(self) -> None:
        self.killed = True
