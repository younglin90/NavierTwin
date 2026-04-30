"""GUI project-open path tests for customer-facing .ntwin workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pyvista = pytest.importorskip("pyvista", reason="pyvista is required for .ntwin GUI tests")
pytest.importorskip("h5py", reason="h5py is required for .ntwin GUI tests")


def _make_dataset() -> object:
    """Create a minimal CFDDataset that round-trips through .ntwin."""
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cells = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    cell_types = np.array([10], dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    mesh.point_data["p"] = np.arange(mesh.n_points, dtype=np.float32)
    return CFDDataset(mesh=mesh, time_steps=[0.0], field_names=["p"])


def _write_ntwin(tmp_path: Path) -> Path:
    from naviertwin.core.export.ntwin_format import save_dataset

    path = tmp_path / "customer_project.ntwin"
    save_dataset(_make_dataset(), path)
    return path


def test_export_panel_load_project_path_emits_project_loaded(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.panels.export_panel import ExportPanel

    path = _write_ntwin(tmp_path)
    panel = ExportPanel()
    qtbot.addWidget(panel)
    emitted: list[tuple[object, object | None]] = []
    panel.project_loaded.connect(lambda dataset, engine: emitted.append((dataset, engine)))

    assert panel.load_project_path(path) is True

    assert panel._dataset is not None
    assert panel._path_edit.text() == str(path)
    assert len(emitted) == 1
    assert emitted[0][0].n_points == 4  # type: ignore[union-attr]
    assert emitted[0][1] is None


def test_export_panel_load_project_path_reports_corrupt_project_without_mutating_state(
    qtbot,
    tmp_path: Path,
) -> None:
    from naviertwin.gui.panels.export_panel import ExportPanel

    valid_path = _write_ntwin(tmp_path)
    corrupt_path = tmp_path / "corrupt_project.ntwin"
    corrupt_path.write_bytes(b"not an hdf5 naviertwin project")
    panel = ExportPanel()
    qtbot.addWidget(panel)

    assert panel.load_project_path(valid_path) is True
    previous_dataset = panel._dataset

    assert panel.load_project_path(corrupt_path) is False

    assert panel._dataset is previous_dataset
    assert panel._path_edit.text() == str(valid_path)
    assert panel.last_project_load_error()


def test_main_window_open_selected_ntwin_routes_to_project_loader(
    qtbot, tmp_path: Path
) -> None:
    from naviertwin.gui.main_window import MainWindow

    path = _write_ntwin(tmp_path)
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    assert win._open_selected_path(path) is True

    assert win._latest_dataset is not None
    assert win._export_panel._dataset is not None
    assert win._export_panel._path_edit.text() == str(path)
    assert win._import_panel._path_edit.text() == ""
    assert "프로젝트 로드 완료" in win._status_label.text()


def test_main_window_open_selected_corrupt_ntwin_surfaces_error(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow
    from naviertwin.utils.config import load_config

    corrupt_path = tmp_path / "corrupt_project.ntwin"
    corrupt_path.write_bytes(b"not an hdf5 naviertwin project")
    cfg_path = tmp_path / "cfg.json"
    warnings: list[tuple[str, str]] = []

    def capture_warning(parent: object, title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(QMessageBox, "warning", capture_warning)
    win = MainWindow(confirm_on_close=False, config_path=cfg_path)
    qtbot.addWidget(win)

    assert win._open_selected_path(corrupt_path) is False

    assert warnings
    assert warnings[0][0] == "프로젝트 열기 실패"
    assert corrupt_path.name in warnings[0][1]
    assert win._latest_dataset is None
    assert win._status_label.text() == "프로젝트 열기 실패"
    assert load_config(cfg_path).recent_projects == []


def test_main_window_open_selected_ntwin_partial_engine_restore(
    qtbot,
    tmp_path: Path,
) -> None:
    from naviertwin.gui.main_window import MainWindow

    path = _write_ntwin(tmp_path)
    path.with_suffix(".engine.pkl").write_bytes(b"not a valid TwinEngine pickle")
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    assert win._open_selected_path(path) is True

    assert win._latest_dataset is not None
    assert win._latest_engine is None
    assert "프로젝트 부분 로드 완료" in win._status_label.text()
    assert "TwinEngine 로드 실패" in win._status_label.text()


def test_main_window_open_selected_cfd_routes_to_import_panel(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.main_window import MainWindow

    path = tmp_path / "case.su2"
    path.write_text("% minimal placeholder\n", encoding="utf-8")
    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    assert win._open_selected_path(path) is True

    assert win._import_panel._path_edit.text() == str(path)
    assert win._tabs.currentWidget() is win._import_panel
