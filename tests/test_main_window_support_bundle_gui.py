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
    assert any("지원 번들 점검" in text for text in actions)


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
        acceptance_json: str | Path | None = None,
        acceptance_summary: str | Path | None = None,
    ) -> dict[str, object]:
        assert include_optional is True
        assert zip_bundle is True
        assert acceptance_json is None
        assert acceptance_summary is None
        return {
            "status": "ok",
            "zip_path": "support-bundle.zip",
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
    assert str(tmp_path / "support-bundle.zip") in messages[0][1]
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
        acceptance_json: str | Path | None = None,
        acceptance_summary: str | Path | None = None,
    ) -> dict[str, object]:
        captured["preflight"] = preflight
        captured["acceptance_json"] = acceptance_json
        captured["acceptance_summary"] = acceptance_summary
        return {
            "status": "ok",
            "zip_path": "support-bundle.zip",
            "files": ["doctor.json", "preflight.json", "metadata.json"],
        }

    monkeypatch.setattr(support_bundle, "build_support_bundle", fake_build_support_bundle)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.information", lambda *args: None)
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    win._import_panel._path_edit.setText(str(case_path))

    win._create_support_bundle_path(tmp_path)

    assert captured["preflight"] == case_path
    assert captured["acceptance_json"] is None
    assert captured["acceptance_summary"] is None


def test_support_bundle_includes_recent_acceptance_artifacts(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import naviertwin.utils.support_bundle as support_bundle
    from naviertwin.gui.main_window import MainWindow

    acceptance_json = tmp_path / "acceptance.json"
    acceptance_json.write_text('{"status": "ok"}\n', encoding="utf-8")
    acceptance_summary = tmp_path / "acceptance.md"
    acceptance_summary.write_text("# Acceptance\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_build_support_bundle(
        outdir: str | Path,
        preflight: str | Path | None = None,
        include_optional: bool = False,
        zip_bundle: bool = False,
        acceptance_json: str | Path | None = None,
        acceptance_summary: str | Path | None = None,
    ) -> dict[str, object]:
        captured["acceptance_json"] = acceptance_json
        captured["acceptance_summary"] = acceptance_summary
        return {
            "status": "ok",
            "zip_path": "support-bundle.zip",
            "files": ["doctor.json", "acceptance.json", "acceptance.md", "metadata.json"],
        }

    monkeypatch.setattr(support_bundle, "build_support_bundle", fake_build_support_bundle)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.information", lambda *args: None)
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    win._last_acceptance_json = acceptance_json
    win._last_acceptance_summary = acceptance_summary

    win._create_support_bundle_path(tmp_path)

    assert captured["acceptance_json"] == acceptance_json
    assert captured["acceptance_summary"] == acceptance_summary


def test_support_bundle_dialog_can_attach_acceptance_artifacts(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    import naviertwin.utils.support_bundle as support_bundle
    from naviertwin.gui.main_window import MainWindow

    acceptance_json = tmp_path / "external-acceptance.json"
    acceptance_json.write_text('{"status": "ok"}\n', encoding="utf-8")
    acceptance_summary = tmp_path / "external-acceptance.md"
    acceptance_summary.write_text("# Acceptance\n", encoding="utf-8")
    captured: dict[str, object] = {}
    open_files = iter([str(acceptance_json), str(acceptance_summary)])

    def fake_build_support_bundle(
        outdir: str | Path,
        preflight: str | Path | None = None,
        include_optional: bool = False,
        zip_bundle: bool = False,
        acceptance_json: str | Path | None = None,
        acceptance_summary: str | Path | None = None,
    ) -> dict[str, object]:
        captured["outdir"] = outdir
        captured["acceptance_json"] = acceptance_json
        captured["acceptance_summary"] = acceptance_summary
        return {
            "status": "ok",
            "zip_path": "support-bundle.zip",
            "files": ["doctor.json", "acceptance.json", "acceptance.md", "metadata.json"],
        }

    monkeypatch.setattr(support_bundle, "build_support_bundle", fake_build_support_bundle)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (next(open_files), ""))
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._create_support_bundle()

    assert captured["outdir"] == tmp_path
    assert captured["acceptance_json"] == acceptance_json
    assert captured["acceptance_summary"] == acceptance_summary


def test_support_bundle_acceptance_selector_skips_prompt_when_recent_exists(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow

    acceptance_json = tmp_path / "acceptance.json"
    acceptance_json.write_text('{"status": "ok"}\n', encoding="utf-8")
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    win._last_acceptance_json = acceptance_json
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args: pytest.fail("recent acceptance should not prompt"),
    )

    assert win._select_support_bundle_acceptance_artifacts() == (None, None)


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


def test_inspect_support_bundle_action_surfaces_success(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    import naviertwin.utils.support_bundle as support_bundle
    from naviertwin.gui.main_window import MainWindow

    bundle = tmp_path / "support-bundle.zip"
    bundle.write_bytes(b"placeholder")
    monkeypatch.setattr(
        support_bundle,
        "inspect_support_bundle",
        lambda path: {
            "status": "ok",
            "kind": "zip",
            "metadata": {"status": "warn", "schema_version": 2},
            "manifest": {"verified": True},
            "artifacts": {"verified": True},
            "warnings": [],
            "errors": [],
        },
    )
    monkeypatch.setattr(
        support_bundle,
        "format_support_bundle_inspection",
        lambda report: "NavierTwin support bundle: ok",
    )
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda parent, title, text: messages.append((title, text)),
    )
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._inspect_support_bundle_path(bundle)

    assert messages == [("지원 번들 점검 완료", "NavierTwin support bundle: ok")]
    assert win._status_label.text() == "지원 번들 점검: ok"


def test_inspect_support_bundle_dialog_selects_zip(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QFileDialog

    from naviertwin.gui.main_window import MainWindow

    bundle = tmp_path / "support-bundle.zip"
    bundle.write_bytes(b"placeholder")
    selected: list[Path] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(bundle), ""),
    )
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    monkeypatch.setattr(win, "_inspect_support_bundle_path", lambda path: selected.append(path))

    win._inspect_support_bundle()

    assert selected == [bundle]


def test_inspect_support_bundle_action_surfaces_failure(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    import naviertwin.utils.support_bundle as support_bundle
    from naviertwin.gui.main_window import MainWindow

    bundle = tmp_path / "support-bundle.zip"
    bundle.write_bytes(b"placeholder")
    monkeypatch.setattr(
        support_bundle,
        "inspect_support_bundle",
        lambda path: {
            "status": "error",
            "kind": "zip",
            "metadata": {"present": False},
            "manifest": {"verified": False},
            "artifacts": {"verified": False},
            "warnings": [],
            "errors": ["missing metadata.json"],
        },
    )
    monkeypatch.setattr(
        support_bundle,
        "format_support_bundle_inspection",
        lambda report: "missing metadata.json",
    )
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, text: warnings.append((title, text)),
    )
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._inspect_support_bundle_path(bundle)

    assert warnings == [("지원 번들 점검 실패", "missing metadata.json")]
    assert win._status_label.text() == "지원 번들 점검: error"
