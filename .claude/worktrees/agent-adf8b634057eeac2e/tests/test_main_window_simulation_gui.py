"""GUI tests for MainWindow Simulation tab integration."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


def test_simulation_is_not_a_workflow_tab(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    assert win._simulation_panel is not None
    tab_texts = [win._tabs.tabText(i) for i in range(win._tabs.count())]
    assert not any("Simulation" in text or "시뮬레이션" in text for text in tab_texts)


def test_library_route_reports_removed_simulation_tab(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._on_library_navigate("Simulation")

    assert "제거" in win._status_label.text()


def test_simulation_burgers_result_routes_to_viewer(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    viewer = _CaptureViewer()
    win._global_viewer = viewer
    result = {
        "U": np.zeros((3, 8), dtype=float),
        "summary": "burgers ok",
    }

    win._on_simulation_done("burgers", result)

    assert viewer.trajectories
    assert viewer.trajectories[0][1] == "u"
    assert "시뮬레이션 완료: burgers" in win._status_label.text()


def test_simulation_lbm_result_routes_to_viewer(qtbot) -> None:
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    viewer = _CaptureViewer()
    win._global_viewer = viewer
    result = {
        "snapshots": np.zeros((2, 4, 5, 3), dtype=float),
        "summary": "lbm ok",
    }

    win._on_simulation_done("lbm_cavity", result)

    assert viewer.fields
    field, name = viewer.fields[0]
    assert field.shape == (2, 4, 5)
    assert name == "ux"


class _CaptureViewer:
    def __init__(self) -> None:
        self.fields: list[tuple[np.ndarray, str]] = []
        self.trajectories: list[tuple[np.ndarray, str]] = []

    def load_field_grid_2d(self, scalar_field: np.ndarray, field_name: str) -> None:
        self.fields.append((scalar_field, field_name))

    def load_1d_trajectory(self, U: np.ndarray, field_name: str) -> None:
        self.trajectories.append((U, field_name))
