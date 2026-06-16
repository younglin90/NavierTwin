"""PyVista viewer wiring and data adaptation tests."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pv = pytest.importorskip("pyvista")


def _make_dataset() -> object:
    from naviertwin.core.cfd_reader.base import CFDDataset

    mesh = pv.Cube().triangulate()
    pressure = np.linspace(0.0, 1.0, mesh.n_points)
    velocity = np.column_stack(
        [
            np.ones(mesh.n_points),
            np.linspace(0.0, 0.5, mesh.n_points),
            np.zeros(mesh.n_points),
        ]
    )
    mesh.point_data["p"] = pressure
    mesh.point_data["U"] = velocity
    return CFDDataset(
        mesh=mesh,
        time_steps=[0.0, 1.0],
        field_names=["p", "U"],
        metadata={
            "time_series_fields": {
                "p": np.vstack([pressure, pressure + 1.0]),
                "U": np.stack([velocity, velocity * 2.0]),
            }
        },
    )


def test_headless_qt_uses_non_interactive_backend(qtbot) -> None:
    from naviertwin.gui.widgets.vtk_viewer import VtkViewer, _qt_runtime_is_headless

    assert _qt_runtime_is_headless()
    viewer = VtkViewer()
    qtbot.addWidget(viewer)

    assert viewer._backend in {"static", "none"}
    assert viewer._plotter is None


def test_load_dataset_updates_controls_and_timestep(qtbot) -> None:
    from naviertwin.gui.widgets.vtk_viewer import VtkViewer

    viewer = VtkViewer()
    qtbot.addWidget(viewer)
    viewer.load_dataset(_make_dataset())

    assert viewer._field_combo.findText("p") >= 0
    assert viewer._field_combo.findText("U") >= 0
    assert viewer._ts_slider.maximum() == 1
    assert viewer._reset_btn.isEnabled()


def test_vector_field_is_rendered_as_magnitude(qtbot) -> None:
    from naviertwin.gui.widgets.vtk_viewer import VtkViewer

    viewer = VtkViewer()
    qtbot.addWidget(viewer)
    viewer.load_dataset(_make_dataset())
    viewer.show_field("U")

    mesh, scalar_name = viewer._prepare_display_mesh()

    assert scalar_name == "U__mag"
    assert "U__mag" in mesh.point_data
    assert np.asarray(mesh.point_data["U__mag"]).ndim == 1


def test_simulation_grid_and_trajectory_loaders(qtbot) -> None:
    from naviertwin.gui.widgets.vtk_viewer import VtkViewer

    viewer = VtkViewer()
    qtbot.addWidget(viewer)
    viewer.load_field_grid_2d(np.ones((2, 4, 5)), field_name="ux")

    assert viewer._dataset is not None
    assert viewer._dataset.n_time_steps == 2
    assert viewer._field_combo.currentText() == "ux"

    viewer.load_1d_trajectory(np.ones((3, 8)), field_name="u")

    assert viewer._dataset is not None
    assert viewer._field_combo.currentText() == "u"


def test_camera_state_preserves_zoom_and_orientation(qtbot) -> None:
    from naviertwin.gui.widgets.vtk_viewer import VtkViewer

    viewer = VtkViewer()
    qtbot.addWidget(viewer)
    viewer._plotter = _FakePlotter()

    state = viewer._capture_camera_state()

    viewer._plotter.camera.position = (10.0, 10.0, 10.0)
    viewer._plotter.camera.focal_point = (0.0, 0.0, 0.0)
    viewer._plotter.camera.up = (0.0, 0.0, 1.0)
    viewer._plotter.camera.parallel_scale = 99.0
    viewer._plotter.camera.clipping_range = (1.0, 2.0)
    viewer._restore_camera_state(state)

    assert viewer._plotter.camera.position == (1.0, 2.0, 3.0)
    assert viewer._plotter.camera.focal_point == (0.1, 0.2, 0.3)
    assert viewer._plotter.camera.up == (0.0, 1.0, 0.0)
    assert viewer._plotter.camera.parallel_scale == 0.42
    assert viewer._plotter.camera.clipping_range == (0.01, 100.0)


def test_main_window_has_simulation_handler() -> None:
    from naviertwin.gui.main_window import MainWindow

    assert hasattr(MainWindow, "_on_simulation_done")


class _FakeCamera:
    position = (1.0, 2.0, 3.0)
    focal_point = (0.1, 0.2, 0.3)
    up = (0.0, 1.0, 0.0)
    parallel_scale = 0.42
    clipping_range = (0.01, 100.0)


class _FakePlotter:
    def __init__(self) -> None:
        self.camera = _FakeCamera()
        self.camera_position = (
            self.camera.position,
            self.camera.focal_point,
            self.camera.up,
        )
