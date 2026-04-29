"""GUI tests for support bundle generation from the Help menu."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")


def test_help_menu_exposes_support_bundle_action(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    actions = [
        action.text()
        for action in win._help_menu.actions()
        if not action.isSeparator()
    ]
    assert any("지원 번들" in text for text in actions)


def test_support_bundle_action_surfaces_success(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    import naviertwin.utils.support_bundle as support_bundle
    from naviertwin.gui.main_window import MainWindow

    def fake_build_support_bundle(
        outdir: str | Path,
        preflight: str | Path | None = None,
        include_optional: bool = False,
        zip_bundle: bool = False,
    ) -> dict[str, object]:
        assert include_optional is True
        assert zip_bundle is True
        return {
            "status": "ok",
            "zip_path": str(Path(outdir) / "support-bundle.zip"),
            "files": ["doctor.json", "metadata.json"],
        }

    monkeypatch.setattr(support_bundle, "build_support_bundle", fake_build_support_bundle)
    messages: list[tuple[str, str]] = []

    def capture_information(parent: object, title: str, text: str) -> None:
        messages.append((title, text))

    monkeypatch.setattr(QMessageBox, "information", capture_information)
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._create_support_bundle_path(tmp_path)

    assert messages
    assert messages[0][0] == "지원 번들 생성 완료"
    assert "support-bundle.zip" in messages[0][1]
    assert win._status_label.text() == "지원 번들 생성: ok"


def test_support_bundle_includes_current_import_path_preflight(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import naviertwin.utils.support_bundle as support_bundle
    from naviertwin.gui.main_window import MainWindow

    case_path = tmp_path / "case.su2"
    case_path.write_text("% placeholder\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_build_support_bundle(
        outdir: str | Path,
        preflight: str | Path | None = None,
        include_optional: bool = False,
        zip_bundle: bool = False,
    ) -> dict[str, object]:
        captured["preflight"] = preflight
        return {
            "status": "ok",
            "zip_path": str(Path(outdir) / "support-bundle.zip"),
            "files": ["doctor.json", "preflight.json", "metadata.json"],
        }

    monkeypatch.setattr(support_bundle, "build_support_bundle", fake_build_support_bundle)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.information", lambda *args: None)
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    win._import_panel._path_edit.setText(str(case_path))

    win._create_support_bundle_path(tmp_path)

    assert captured["preflight"] == case_path


def test_support_bundle_action_surfaces_errors(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    import naviertwin.utils.support_bundle as support_bundle
    from naviertwin.gui.main_window import MainWindow

    def fail_build_support_bundle(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("disk full")

    monkeypatch.setattr(support_bundle, "build_support_bundle", fail_build_support_bundle)
    warnings: list[tuple[str, str]] = []

    def capture_warning(parent: object, title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(QMessageBox, "warning", capture_warning)
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._create_support_bundle_path(tmp_path)

    assert warnings
    assert warnings[0][0] == "지원 번들 생성 실패"
    assert "disk full" in warnings[0][1]
    assert win._status_label.text() == "지원 번들 생성 실패"
