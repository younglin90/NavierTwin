"""GUI package root public API tests."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def test_gui_root_exposes_main_window_lazily() -> None:
    """Top-level GUI package should expose the shipped desktop entrypoint."""
    pytest.importorskip("PySide6")

    import naviertwin.gui as gui
    from naviertwin.gui.main_window import MainWindow

    assert "MainWindow" in gui.__all__
    assert gui.MainWindow is MainWindow


def test_gui_root_does_not_eagerly_import_pyside() -> None:
    """Importing naviertwin.gui should stay cheap until Qt entrypoints are used."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import sys; "
        "import naviertwin.gui; "
        "raise SystemExit(1 if 'PySide6' in sys.modules else 0)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert completed.returncode == 0


@pytest.mark.parametrize(
    ("package_name", "expected"),
    [
        (
            "naviertwin.gui.panels",
            {
                "ImportPanel": "naviertwin.gui.panels.import_panel",
                "AnalyzePanel": "naviertwin.gui.panels.analyze_panel",
                "ReducePanel": "naviertwin.gui.panels.reduce_panel",
                "ModelPanel": "naviertwin.gui.panels.model_panel",
                "TwinPanel": "naviertwin.gui.panels.twin_panel",
                "ExplainabilityPanel": "naviertwin.gui.panels.explainability_panel",
                "ExportPanel": "naviertwin.gui.panels.export_panel",
                "SimulationPanel": "naviertwin.gui.panels.simulation_panel",
                "PostProcessPanel": "naviertwin.gui.panels.postproc_panel",
            },
        ),
        (
            "naviertwin.gui.widgets",
            {
                "VtkViewer": "naviertwin.gui.widgets.vtk_viewer",
                "ModelCompareWidget": "naviertwin.gui.widgets.model_compare_widget",
                "LossCurveWidget": "naviertwin.gui.widgets.loss_curve_widget",
            },
        ),
    ],
)
def test_gui_subpackage_roots_export_shipped_surface(
    package_name: str,
    expected: dict[str, str],
) -> None:
    """Panel and widget package roots should match customer-facing GUI classes."""
    pytest.importorskip("PySide6")

    package = import_module(package_name)

    assert len(package.__all__) == len(set(package.__all__))
    assert set(expected).issubset(set(package.__all__))
    for symbol, module_name in expected.items():
        source_module = import_module(module_name)
        assert getattr(package, symbol) is getattr(source_module, symbol)
