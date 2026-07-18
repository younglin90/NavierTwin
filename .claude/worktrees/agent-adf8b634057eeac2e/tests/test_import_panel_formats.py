"""ImportPanel supported-format visibility tests."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


class TestImportPanelFormatVisibility:
    def test_file_dialog_filter_includes_registered_reader_extensions(self) -> None:
        from naviertwin.core.cfd_reader import ReaderFactory
        from naviertwin.gui.panels.import_panel import cfd_file_filter

        file_filter = cfd_file_filter()
        missing = [
            ext for ext in ReaderFactory.registered_extensions()
            if f"*{ext}" not in file_filter
        ]

        assert missing == []

    def test_main_window_open_filter_includes_registered_readers_and_projects(self) -> None:
        from naviertwin.core.cfd_reader import ReaderFactory
        from naviertwin.gui.main_window import open_file_filter

        file_filter = open_file_filter()
        missing = [
            ext for ext in ReaderFactory.registered_extensions()
            if f"*{ext}" not in file_filter
        ]

        assert missing == []
        assert "*.ntwin" in file_filter

    def test_supported_format_label_mentions_commercial_formats(self) -> None:
        from naviertwin.gui.panels.import_panel import supported_format_label

        label = supported_format_label()
        for name in ["PVD", "Fluent", "CGNS", "Gmsh", "SU2"]:
            assert name in label

    def test_import_panel_subtitle_uses_full_supported_format_label(self, qtbot) -> None:
        from naviertwin.gui.panels.import_panel import (
            ImportPanel,
            supported_format_label,
        )

        panel = ImportPanel()
        qtbot.addWidget(panel)
        assert supported_format_label() in [
            child.text() for child in panel.findChildren(type(panel._status_label))
        ]

    def test_import_panel_selects_multiple_cfd_cases(
        self,
        qtbot,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from PySide6.QtWidgets import QFileDialog

        from naviertwin.gui.panels.import_panel import ImportPanel

        case0 = tmp_path / "case_000.vtu"
        case1 = tmp_path / "case_001.vtu"
        case0.write_text("", encoding="utf-8")
        case1.write_text("", encoding="utf-8")
        panel = ImportPanel()
        qtbot.addWidget(panel)
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileNames",
            lambda *_, **__: ([str(case0), str(case1)], ""),
        )

        panel._browse_cases()

        assert panel._case_paths == [case0, case1]
        assert panel._current_path == case0
        assert panel._path_edit.text() == str(case0)
        assert panel._load_btn.isEnabled() is True
        assert "2개 케이스" in panel._case_label.text()
