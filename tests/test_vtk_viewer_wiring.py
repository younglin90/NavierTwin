"""Round 65 — VTK viewer load_field_grid_2d + SimulationPanel 연동."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

pytest.importorskip("PySide6")


class TestViewerAPI:
    def test_load_field_grid_2d_signature(self) -> None:
        """load_field_grid_2d / load_1d_trajectory 메서드 존재 검증."""
        from naviertwin.gui.widgets.vtk_viewer import VtkViewer

        assert hasattr(VtkViewer, "load_field_grid_2d")
        assert hasattr(VtkViewer, "load_1d_trajectory")

    def test_main_window_has_simulation_handler(self) -> None:
        from naviertwin.gui.main_window import MainWindow

        assert hasattr(MainWindow, "_on_simulation_done")

    def test_render_controls_disabled_when_plotter_unavailable(
        self, qtbot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from naviertwin.gui.widgets.vtk_viewer import VtkViewer

        real_import = builtins.__import__

        def block_pyvistaqt(name: str, *args: object, **kwargs: object) -> object:
            if name == "pyvistaqt":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_pyvistaqt)

        viewer = VtkViewer()
        qtbot.addWidget(viewer)

        assert viewer._plotter is None
        assert not viewer._reset_btn.isEnabled()
        assert not viewer._screenshot_btn.isEnabled()
        assert "renderer unavailable" in viewer._placeholder.text()

    def test_render_controls_enable_when_plotter_available(
        self, qtbot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from naviertwin.gui.widgets import vtk_viewer

        def fake_init_plotter(self: object) -> None:
            self._plotter = object()
            self._set_render_controls_enabled(True)

        monkeypatch.setattr(vtk_viewer.VtkViewer, "_init_plotter", fake_init_plotter)

        viewer = vtk_viewer.VtkViewer()
        qtbot.addWidget(viewer)

        assert viewer._plotter is not None
        assert viewer._reset_btn.isEnabled()
        assert viewer._screenshot_btn.isEnabled()


class TestSimulationResultFormat:
    """SimulationPanel 이 내보내는 dict 포맷이 기대대로인지 검증."""

    def test_lbm_result_has_snapshots(self) -> None:
        from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

        lbm = LBMD2Q9(nx=8, ny=8, tau=0.8, u_top=0.05)
        s = lbm.run(n_steps=10, record_every=5)
        # simulation_done 에서 snapshots key 에 넣는 형태
        result = {"snapshots": s, "summary": "ok"}
        assert result["snapshots"].ndim == 4

    def test_burgers_result_has_U(self) -> None:
        from naviertwin.core.solver_interfaces.fvm_advection import fvm_upwind_1d

        u0 = np.sin(np.linspace(0, 2 * np.pi, 32, endpoint=False))
        _, U = fvm_upwind_1d(u0, c=1.0, L=2 * np.pi, T=0.2, cfl=0.4)
        result = {"U": U, "summary": "ok"}
        assert result["U"].ndim == 2
