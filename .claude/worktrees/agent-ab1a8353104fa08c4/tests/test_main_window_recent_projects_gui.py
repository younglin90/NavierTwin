"""GUI tests for recent .ntwin project persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")


def _make_dataset() -> object:
    pyvista = pytest.importorskip("pyvista")
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
    mesh = pyvista.UnstructuredGrid(cells, cell_types, points)
    mesh.point_data["p"] = np.arange(mesh.n_points, dtype=np.float32)
    return CFDDataset(mesh=mesh, time_steps=[0.0], field_names=["p"])


def _write_ntwin(tmp_path: Path) -> Path:
    pytest.importorskip("h5py")
    from naviertwin.core.export.ntwin_format import save_dataset

    path = tmp_path / "recent_project.ntwin"
    save_dataset(_make_dataset(), path)
    return path


def test_recent_projects_menu_empty_state(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False, config_path=tmp_path / "cfg.json")
    qtbot.addWidget(win)

    actions = win._recent_projects_menu.actions()
    assert len(actions) == 1
    assert actions[0].text() == "최근 프로젝트 없음"
    assert actions[0].isEnabled() is False


def test_recent_projects_dedupes_limits_and_persists(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.main_window import MainWindow
    from naviertwin.utils.config import load_config

    cfg_path = tmp_path / "cfg.json"
    win = MainWindow(confirm_on_close=False, config_path=cfg_path)
    qtbot.addWidget(win)
    paths = [tmp_path / f"project_{i}.ntwin" for i in range(12)]

    for path in paths:
        win._remember_recent_project(path)
    win._remember_recent_project(paths[5])

    cfg = load_config(cfg_path)
    assert len(cfg.recent_projects) == 10
    assert cfg.recent_projects[0] == str(paths[5].resolve())
    assert len(set(cfg.recent_projects)) == 10
    assert win._recent_projects_menu.actions()[0].data() == str(paths[5].resolve())


def test_open_selected_ntwin_records_recent_project(qtbot, tmp_path: Path) -> None:
    from naviertwin.gui.main_window import MainWindow
    from naviertwin.utils.config import load_config

    path = _write_ntwin(tmp_path)
    cfg_path = tmp_path / "cfg.json"
    win = MainWindow(confirm_on_close=False, config_path=cfg_path)
    qtbot.addWidget(win)

    win._open_selected_path(path)

    cfg = load_config(cfg_path)
    assert cfg.recent_projects[0] == str(path.resolve())
    assert win._recent_projects_menu.actions()[0].text() == path.name


def test_corrupt_recent_project_is_removed_after_failed_open(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from naviertwin.gui.main_window import MainWindow
    from naviertwin.utils.config import load_config

    path = tmp_path / "corrupt_recent.ntwin"
    path.write_bytes(b"not an hdf5 naviertwin project")
    cfg_path = tmp_path / "cfg.json"
    win = MainWindow(confirm_on_close=False, config_path=cfg_path)
    qtbot.addWidget(win)
    win._remember_recent_project(path)

    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    action = win._recent_projects_menu.actions()[0]
    action.trigger()

    assert load_config(cfg_path).recent_projects == []
    assert win._recent_projects_menu.actions()[0].text() == "최근 프로젝트 없음"
