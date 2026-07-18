"""GUI smoke tests for advanced AnalyzePanel diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
Qt = pytest.importorskip("PySide6.QtCore").Qt
pyvista = pytest.importorskip("pyvista", reason="pyvista is required for AnalyzePanel tests")


def _make_time_series_dataset() -> object:
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    xs = np.linspace(0.0, 1.0, 4)
    ys = np.linspace(0.0, 1.0, 4)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float32)
    mesh = pv.PolyData(points)
    velocity = np.column_stack([
        1.0 + points[:, 1],
        0.2 * points[:, 0],
        np.zeros(len(points)),
    ]).astype(np.float32)
    mesh.point_data["U"] = velocity

    time_steps = np.linspace(0.0, 0.31, 32)
    series = np.stack([
        velocity * (1.0 + 0.05 * np.sin(2.0 * np.pi * t / 0.32))
        for t in time_steps
    ]).astype(np.float32)
    return CFDDataset(
        mesh=mesh,
        time_steps=list(time_steps),
        field_names=["U"],
        metadata={"time_series_fields": {"U": series}},
    )


def test_analyze_panel_lists_advanced_customer_diagnostics(qtbot) -> None:
    from naviertwin.gui.panels.analyze_panel import AnalyzePanel, analysis_method_labels

    panel = AnalyzePanel()
    qtbot.addWidget(panel)

    labels = analysis_method_labels()
    for label in [
        "SPOD (Modal)",
        "SINDy (Equation Discovery)",
        "Wavelet / STFT",
        "Boundary Layer Thickness",
        "Nondimensional Numbers",
        "FTLE / LCS Quick Check",
        "PGD 3D Quick Decomposition",
        "Entropy Generation 2D",
    ]:
        assert label in labels
        assert panel._method_list.findItems(label, Qt.MatchFlag.MatchExactly)


def test_advanced_analyze_dispatch_runs_core_functions(qtbot) -> None:
    from naviertwin.gui.panels.analyze_panel import AnalyzePanel

    panel = AnalyzePanel()
    qtbot.addWidget(panel)
    panel.set_dataset(_make_time_series_dataset())

    expected_tokens = {
        "spod": "SPOD:",
        "sindy": "SINDy:",
        "wavelet": "STFT:",
        "boundary_layer": "Boundary Layer:",
        "nondim": "Nondim:",
        "ftle": "FTLE:",
        "pgd": "PGD:",
        "entropy_generation": "Entropy Generation:",
    }
    for method, token in expected_tokens.items():
        result = panel._dispatch(method)
        assert token in str(result)


def test_analyze_panel_sindy_discovers_equation(qtbot) -> None:
    from naviertwin.gui.panels.analyze_panel import AnalyzePanel

    panel = AnalyzePanel()
    qtbot.addWidget(panel)
    panel.set_dataset(_make_sindy_dataset())

    result = str(panel._dispatch("sindy"))

    assert "SINDy:" in result
    assert "dx0/dt" in result
    assert "x0" in result


def _make_sindy_dataset() -> object:
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    time_steps = np.linspace(0.0, 3.0, 40)
    signal = np.exp(-0.5 * time_steps)
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    mesh = pv.PolyData(points)
    mesh.point_data["u"] = np.array([signal[0]], dtype=np.float32)
    series = signal.reshape(-1, 1).astype(np.float32)
    return CFDDataset(
        mesh=mesh,
        time_steps=list(time_steps),
        field_names=["u"],
        metadata={"time_series_fields": {"u": series}},
    )
