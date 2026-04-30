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
    assert any("벤치마크" in text for text in actions)
    assert any("파이프라인 데모" in text for text in actions)
    assert any("CSV 스냅샷으로 트윈 생성" in text for text in actions)
    assert any("저장된 트윈 예측" in text for text in actions)
    assert any("배포 트윈 디렉토리 예측" in text for text in actions)
    assert any("저장된 트윈 검증" in text for text in actions)
    assert any("배포 트윈 디렉토리 검증" in text for text in actions)
    assert any("트윈 산출물 패키징" in text for text in actions)
    assert any("트윈 패키지 정보 보기" in text for text in actions)
    assert any("트윈 패키지 검증" in text for text in actions)
    assert any("트윈 패키지 검증 후 추출" in text for text in actions)
    assert any("API 서버 시작" in text for text in actions)
    assert any("API 서버 중지" in text for text in actions)


def test_benchmark_action_runs_cli_and_surfaces_result(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    calls: list[str] = []
    messages: list[tuple[str, str]] = []

    monkeypatch.setattr(win, "_run_benchmark_cli", lambda kind: calls.append(kind) or 0)
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    win._run_benchmark()

    assert calls == ["burgers"]
    assert messages
    assert messages[0][0] == "벤치마크 완료"
    assert win._status_label.text() == "벤치마크 완료: burgers"


def test_benchmark_action_surfaces_failure(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    warnings: list[tuple[str, str]] = []

    monkeypatch.setattr(win, "_run_benchmark_cli", lambda kind: 1)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )

    win._run_benchmark()

    assert warnings
    assert warnings[0][0] == "벤치마크 실패"
    assert "1" in warnings[0][1]
    assert win._status_label.text() == "벤치마크 실패"


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


def test_build_twin_from_csv_paths_runs_cli_and_surfaces_result(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    csv_paths = [tmp_path / "snap_0.csv", tmp_path / "snap_1.csv"]
    calls: list[tuple[list[Path], str, Path]] = []
    messages: list[tuple[str, str]] = []

    monkeypatch.setattr(
        win,
        "_run_build_twin_cli",
        lambda paths, *, field_column, outdir: (
            calls.append((paths, field_column, outdir)) or 0
        ),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    win._build_twin_from_csv_paths(csv_paths, field_column="U", outdir=tmp_path)

    assert calls == [(csv_paths, "U", tmp_path)]
    assert messages
    assert messages[0][0] == "트윈 생성 완료"
    assert win._status_label.text() == "트윈 생성 완료"


def test_build_twin_from_csv_paths_surfaces_failure(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    warnings: list[tuple[str, str]] = []

    monkeypatch.setattr(win, "_run_build_twin_cli", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )

    win._build_twin_from_csv_paths([tmp_path / "snap.csv"], field_column="U", outdir=tmp_path)

    assert warnings
    assert warnings[0][0] == "트윈 생성 실패"
    assert "2" in warnings[0][1]
    assert win._status_label.text() == "트윈 생성 실패"


def test_predict_twin_from_engine_path_runs_cli_and_surfaces_result(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    calls: list[tuple[Path, str, Path | None]] = []
    messages: list[tuple[str, str]] = []

    monkeypatch.setattr(
        win,
        "_run_predict_twin_cli",
        lambda engine_path, *, params, output: (
            calls.append((engine_path, params, output)) or 0
        ),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    engine_path = tmp_path / "engine.pkl"
    output = tmp_path / "prediction.csv"
    win._predict_twin_from_engine_path(engine_path, params="0.25", output=output)

    assert calls == [(engine_path, "0.25", output)]
    assert messages
    assert messages[0][0] == "트윈 예측 완료"
    assert str(output) in messages[0][1]
    assert win._status_label.text() == "트윈 예측 완료"


def test_predict_twin_from_engine_path_surfaces_failure(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    warnings: list[tuple[str, str]] = []

    monkeypatch.setattr(win, "_run_predict_twin_cli", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )

    win._predict_twin_from_engine_path(
        tmp_path / "engine.pkl",
        params="0.25",
        output=None,
    )

    assert warnings
    assert warnings[0][0] == "트윈 예측 실패"
    assert "2" in warnings[0][1]
    assert win._status_label.text() == "트윈 예측 실패"


def test_predict_twin_from_artifacts_dir_path_runs_cli_and_surfaces_result(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    calls: list[tuple[Path, str, Path | None]] = []
    messages: list[tuple[str, str]] = []

    monkeypatch.setattr(
        win,
        "_run_predict_twin_artifacts_cli",
        lambda artifacts_dir, *, params, output: (
            calls.append((artifacts_dir, params, output)) or 0
        ),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    artifacts_dir = tmp_path / "deployed-twin"
    output = tmp_path / "prediction.csv"
    win._predict_twin_from_artifacts_dir_path(artifacts_dir, params="0.25", output=output)

    assert calls == [(artifacts_dir, "0.25", output)]
    assert messages
    assert messages[0][0] == "배포 트윈 예측 완료"
    assert str(output) in messages[0][1]
    assert win._status_label.text() == "배포 트윈 예측 완료"


def test_predict_twin_from_artifacts_dir_path_surfaces_failure(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    warnings: list[tuple[str, str]] = []

    monkeypatch.setattr(win, "_run_predict_twin_artifacts_cli", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )

    win._predict_twin_from_artifacts_dir_path(
        tmp_path / "deployed-twin",
        params="0.25",
        output=None,
    )

    assert warnings
    assert warnings[0][0] == "배포 트윈 예측 실패"
    assert "2" in warnings[0][1]
    assert win._status_label.text() == "배포 트윈 예측 실패"


def test_validate_twin_from_paths_runs_cli_and_surfaces_result(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    csv_paths = [tmp_path / "snap_0.csv", tmp_path / "snap_1.csv"]
    calls: list[tuple[Path, list[Path], str, Path | None, float | None, float | None]] = []
    messages: list[tuple[str, str]] = []

    monkeypatch.setattr(
        win,
        "_run_validate_twin_cli",
        lambda engine_path, paths, *, field_column, output, max_rmse, min_r2, **kwargs: (
            calls.append((engine_path, paths, field_column, output, max_rmse, min_r2)) or 0
        ),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    engine_path = tmp_path / "engine.pkl"
    output = tmp_path / "validation.json"
    win._validate_twin_from_paths(
        engine_path,
        csv_paths,
        field_column="U",
        output=output,
        max_rmse=0.05,
        min_r2=0.98,
    )

    assert calls == [(engine_path, csv_paths, "U", output, 0.05, 0.98)]
    assert messages
    assert messages[0][0] == "트윈 검증 완료"
    assert str(output) in messages[0][1]
    assert win._status_label.text() == "트윈 검증 완료"


def test_validate_twin_from_paths_surfaces_failure(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    warnings: list[tuple[str, str]] = []

    monkeypatch.setattr(win, "_run_validate_twin_cli", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )

    win._validate_twin_from_paths(
        tmp_path / "engine.pkl",
        [tmp_path / "snap.csv"],
        field_column="U",
        output=None,
    )

    assert warnings
    assert warnings[0][0] == "트윈 검증 실패"
    assert "2" in warnings[0][1]
    assert win._status_label.text() == "트윈 검증 실패"


def test_parse_validation_thresholds() -> None:
    from naviertwin.gui.main_window import MainWindow

    assert MainWindow._parse_validation_thresholds("") == (None, None, None)
    assert MainWindow._parse_validation_thresholds("0.05,0.98,0.1") == (
        0.05,
        0.98,
        0.1,
    )
    assert MainWindow._parse_validation_thresholds(",0.99") == (None, 0.99, None)
    with pytest.raises(ValueError):
        MainWindow._parse_validation_thresholds("bad")


def test_validate_twin_from_artifacts_dir_paths_runs_cli_and_surfaces_result(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    artifacts_dir = tmp_path / "deployed-twin"
    csv_paths = [tmp_path / "snap_0.csv", tmp_path / "snap_1.csv"]
    calls: list[tuple[Path, list[Path], str, Path | None, float | None, float | None]] = []
    messages: list[tuple[str, str]] = []

    monkeypatch.setattr(
        win,
        "_run_validate_twin_artifacts_cli",
        lambda root, paths, *, field_column, output, max_rmse, min_r2, **kwargs: (
            calls.append((root, paths, field_column, output, max_rmse, min_r2)) or 0
        ),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    output = tmp_path / "validation.json"
    win._validate_twin_from_artifacts_dir_paths(
        artifacts_dir,
        csv_paths,
        field_column="U",
        output=output,
        max_rmse=0.1,
        min_r2=0.9,
    )

    assert calls == [(artifacts_dir, csv_paths, "U", output, 0.1, 0.9)]
    assert messages
    assert messages[0][0] == "배포 트윈 검증 완료"
    assert str(output) in messages[0][1]
    assert win._status_label.text() == "배포 트윈 검증 완료"


def test_validate_twin_from_artifacts_dir_paths_surfaces_failure(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    warnings: list[tuple[str, str]] = []

    monkeypatch.setattr(win, "_run_validate_twin_artifacts_cli", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )

    win._validate_twin_from_artifacts_dir_paths(
        tmp_path / "deployed-twin",
        [tmp_path / "snap.csv"],
        field_column="U",
        output=None,
    )

    assert warnings
    assert warnings[0][0] == "배포 트윈 검증 실패"
    assert "2" in warnings[0][1]
    assert win._status_label.text() == "배포 트윈 검증 실패"


def test_package_twin_from_paths_runs_cli_and_surfaces_result(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    calls: list[tuple[Path, Path, Path | None]] = []
    messages: list[tuple[str, str]] = []

    monkeypatch.setattr(
        win,
        "_run_package_twin_cli",
        lambda artifacts_dir, *, output, include_validation: (
            calls.append((artifacts_dir, output, include_validation)) or 0
        ),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    output = tmp_path / "delivery.zip"
    validation = tmp_path / "validation.json"
    win._package_twin_from_paths(
        tmp_path / "twin",
        output=output,
        include_validation=validation,
    )

    assert calls == [(tmp_path / "twin", output, validation)]
    assert messages
    assert messages[0][0] == "트윈 패키징 완료"
    assert str(output) in messages[0][1]
    assert win._status_label.text() == "트윈 패키징 완료"


def test_package_twin_from_paths_surfaces_failure(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    warnings: list[tuple[str, str]] = []

    monkeypatch.setattr(win, "_run_package_twin_cli", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )

    win._package_twin_from_paths(tmp_path / "twin", output=tmp_path / "delivery.zip")

    assert warnings
    assert warnings[0][0] == "트윈 패키징 실패"
    assert "2" in warnings[0][1]
    assert win._status_label.text() == "트윈 패키징 실패"


def test_inspect_twin_package_path_surfaces_metadata(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    calls: list[Path] = []
    messages: list[tuple[str, str]] = []

    monkeypatch.setattr(
        win,
        "_run_inspect_twin_package_cli",
        lambda package_path: calls.append(package_path)
        or {
            "status": "ok",
            "format": "NavierTwin delivery package",
            "manifest_entry_count": 7,
            "validation_included": True,
            "metrics": {"rmse": 0.01, "r2": 0.99},
            "errors": [],
        },
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    package_path = tmp_path / "delivery.zip"
    win._inspect_twin_package_path(package_path)

    assert calls == [package_path]
    assert messages
    assert messages[0][0] == "트윈 패키지 정보 조회 완료"
    assert "NavierTwin delivery package" in messages[0][1]
    assert "RMSE: 0.01" in messages[0][1]
    assert win._status_label.text() == "트윈 패키지 정보 조회 완료"


def test_inspect_twin_package_path_surfaces_failure(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    warnings: list[tuple[str, str]] = []

    monkeypatch.setattr(
        win,
        "_run_inspect_twin_package_cli",
        lambda package_path: {"status": "failed", "errors": ["integrity mismatch"]},
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )

    win._inspect_twin_package_path(tmp_path / "delivery.zip")

    assert warnings
    assert warnings[0][0] == "트윈 패키지 정보 조회 실패"
    assert "integrity mismatch" in warnings[0][1]
    assert win._status_label.text() == "트윈 패키지 정보 조회 실패"


def test_verify_twin_package_path_runs_cli_and_surfaces_result(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    calls: list[Path] = []
    messages: list[tuple[str, str]] = []

    monkeypatch.setattr(
        win,
        "_run_verify_twin_package_cli",
        lambda package_path: calls.append(package_path) or 0,
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    package_path = tmp_path / "delivery.zip"
    win._verify_twin_package_path(package_path)

    assert calls == [package_path]
    assert messages
    assert messages[0][0] == "트윈 패키지 검증 완료"
    assert str(package_path) in messages[0][1]
    assert win._status_label.text() == "트윈 패키지 검증 완료"


def test_verify_twin_package_path_surfaces_failure(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    warnings: list[tuple[str, str]] = []

    monkeypatch.setattr(win, "_run_verify_twin_package_cli", lambda package_path: 1)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )

    win._verify_twin_package_path(tmp_path / "delivery.zip")

    assert warnings
    assert warnings[0][0] == "트윈 패키지 검증 실패"
    assert "1" in warnings[0][1]
    assert win._status_label.text() == "트윈 패키지 검증 실패"


def test_verify_twin_package_path_can_extract_after_verification(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    calls: list[tuple[Path, Path | None]] = []
    messages: list[tuple[str, str]] = []

    def fake_verify(package_path: Path, *, extract_to: Path | None = None) -> int:
        calls.append((package_path, extract_to))
        return 0

    monkeypatch.setattr(win, "_run_verify_twin_package_cli", fake_verify)
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )

    package_path = tmp_path / "delivery.zip"
    extract_to = tmp_path / "deployed"
    win._verify_twin_package_path(package_path, extract_to=extract_to)

    assert calls == [(package_path, extract_to)]
    assert messages
    assert messages[0][0] == "트윈 패키지 검증 및 추출 완료"
    assert str(extract_to) in messages[0][1]
    assert win._status_label.text() == "트윈 패키지 검증 및 추출 완료"


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
    assert fake.finished.callbacks


def test_api_server_finish_updates_status(qtbot) -> None:
    from PySide6.QtCore import QProcess

    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    fake = _FakeProcess(started=True)
    win._server_process = fake

    win._on_api_server_finished(2, QProcess.ExitStatus.NormalExit)

    assert win._server_process is None
    assert win._status_label.text() == "API 서버 종료됨: exit=2"


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

        self.finished = _FakeSignal()
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


class _FakeSignal:
    def __init__(self) -> None:
        self.callbacks: list[object] = []

    def connect(self, callback: object) -> None:
        self.callbacks.append(callback)
