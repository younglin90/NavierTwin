"""GUI tests for Library / capability browser panel."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_library_panel_lists_libraries_and_capabilities(qtbot) -> None:
    from naviertwin.gui.panels.library_panel import LibraryPanel

    panel = LibraryPanel()
    qtbot.addWidget(panel)

    assert panel._library_table.rowCount() > 0
    assert panel._capability_list.count() > 0
    assert any(
        "PhysicsNeMo" in panel._capability_list.item(i).text()
        or "PINN" in panel._capability_list.item(i).text()
        for i in range(panel._capability_list.count())
    )


def test_library_panel_runs_fast_demo(qtbot) -> None:
    from naviertwin.gui.panels.library_panel import LibraryPanel

    panel = LibraryPanel()
    qtbot.addWidget(panel)

    target_row = _find_row(panel, "Fan / Duct / Pump calculators")
    panel._capability_list.setCurrentRow(target_row)

    emitted: list[str] = []
    panel.capability_done.connect(lambda cap_id, _result: emitted.append(cap_id))
    panel._run_selected_demo()

    assert emitted == ["applied.calculators"]
    assert "fan_scaled_Q_H_P" in panel._result_text.toPlainText()


def test_library_panel_recommends_feature_pack_for_torch_capability(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from naviertwin.gui.panels import library_panel
    from naviertwin.gui.panels.library_panel import LibraryPanel

    real_available = library_panel._module_available
    monkeypatch.setattr(
        library_panel,
        "_module_available",
        lambda module: False if module == "torch" else real_available(module),
    )

    panel = LibraryPanel()
    qtbot.addWidget(panel)

    target_row = _find_row(panel, "FNO1D")
    panel._capability_list.setCurrentRow(target_row)

    assert panel._selected_feature_pack_id == "ml-cpu"
    assert "FeaturePack-ml-cpu" in panel._detail_text.toPlainText()


def test_library_panel_navigation_signal(qtbot) -> None:
    from naviertwin.gui.panels.library_panel import LibraryPanel

    panel = LibraryPanel()
    qtbot.addWidget(panel)

    target_row = _find_row(panel, "Post-Tools Facade")
    panel._capability_list.setCurrentRow(target_row)

    routes: list[str] = []
    panel.navigate_requested.connect(routes.append)
    panel._navigate_selected()

    assert routes == ["Post-Tools"]


def test_postprocess_operation_is_exposed(qtbot) -> None:
    from naviertwin.gui.panels.library_panel import LibraryPanel

    panel = LibraryPanel()
    qtbot.addWidget(panel)

    panel._search_edit.setText("psd_welch")

    assert panel._capability_list.count() == 1
    assert "psd_welch" in panel._capability_list.item(0).text()


def test_core_api_inventory_exposes_public_core_surface() -> None:
    from naviertwin.gui.panels.library_panel import list_core_api_items

    items = list_core_api_items()

    assert len(items) > 500
    assert any(
        item.name == "PhysicsNEMOWrapper"
        and item.import_path == "naviertwin.core.physnemo.PhysicsNEMOWrapper"
        for item in items
    )
    assert any(item.name == "PostProcessFacade" for item in items)


def test_library_panel_searches_core_api_table(qtbot) -> None:
    from naviertwin.gui.panels.library_panel import LibraryPanel

    panel = LibraryPanel()
    qtbot.addWidget(panel)

    assert panel._api_table.rowCount() > 500

    panel._search_edit.setText("PhysicsNEMOWrapper")

    assert panel._api_table.rowCount() >= 1
    assert any(
        panel._api_table.item(row, 2).text() == "PhysicsNEMOWrapper"
        for row in range(panel._api_table.rowCount())
    )

    panel._on_api_selected(0, 0)
    assert "import_path" in panel._result_text.toPlainText()


def _find_row(panel: object, needle: str) -> int:
    capability_list = panel._capability_list
    for row in range(capability_list.count()):
        if needle in capability_list.item(row).text():
            return row
    raise AssertionError(f"capability not found: {needle}")
