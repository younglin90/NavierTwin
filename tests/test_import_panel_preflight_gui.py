"""GUI tests for ImportPanel dataset readiness preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvista", reason="pyvista is required for SU2 preflight fixture")

FIXTURES = Path(__file__).parent / "fixtures"
SU2_PATH = FIXTURES / "tiny_square.su2"


def test_import_panel_preflight_button_follows_path_state(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.panels.import_panel import ImportPanel

    panel = ImportPanel()
    qtbot.addWidget(panel)

    assert panel._preflight_btn.isEnabled() is False
    panel._path_edit.setText(str(tmp_path / "case.su2"))

    assert panel._preflight_btn.isEnabled() is True
    assert panel._load_btn.isEnabled() is True


def test_import_panel_preflight_reports_ready_fixture(qtbot) -> None:
    from naviertwin.gui.panels.import_panel import ImportPanel

    panel = ImportPanel()
    qtbot.addWidget(panel)
    panel._path_edit.setText(str(SU2_PATH))

    report = panel._run_preflight_path(SU2_PATH)

    assert report["status"] == "ok"
    assert report["readiness_score"] == 100
    assert panel._status_label.text() == "Preflight: ok (100/100)"
    assert "NavierTwin preflight: ok" in panel._log_text.toPlainText()


def test_import_panel_preflight_reports_missing_path(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.panels.import_panel import ImportPanel

    missing = tmp_path / "missing.su2"
    panel = ImportPanel()
    qtbot.addWidget(panel)

    report = panel._run_preflight_path(missing)

    assert report["status"] == "error"
    assert report["errors"] == ["path_exists"]
    assert panel._status_label.text() == "Preflight: error (0/100)"
    assert "path_exists: error" in panel._log_text.toPlainText()
