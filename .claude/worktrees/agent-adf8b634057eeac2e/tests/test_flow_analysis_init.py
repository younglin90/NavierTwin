"""Flow-analysis package root public API tests."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_flow_analysis_root_exports_core_postproc_helpers() -> None:
    """Package root should expose customer-facing post-processing helpers."""
    import naviertwin.core.flow_analysis as flow_analysis

    expected = {
        "BaseFlowAnalyzer": "naviertwin.core.flow_analysis.base",
        "mean_field": "naviertwin.core.flow_analysis.reynolds_stats",
        "welch_psd": "naviertwin.core.flow_analysis.psd",
        "pressure_force": "naviertwin.core.flow_analysis.surface_integrals",
        "mass_flux": "naviertwin.core.flow_analysis.plane_flux",
        "interp_field": "naviertwin.core.flow_analysis.time_interp",
        "cart_to_cyl": "naviertwin.core.flow_analysis.coord_transform",
        "slice_axis_aligned": "naviertwin.core.flow_analysis.slice_extract",
        "safe_eval": "naviertwin.core.flow_analysis.expression_eval",
        "RunningMoments": "naviertwin.core.flow_analysis.running_moments",
        "savgol_filter": "naviertwin.core.flow_analysis.denoise",
        "percentile": "naviertwin.core.flow_analysis.quantile_stats",
        "eof_decomposition": "naviertwin.core.flow_analysis.eof_analysis",
        "ks_test_normal": "naviertwin.core.flow_analysis.goodness_of_fit",
        "trigger_average": "naviertwin.core.flow_analysis.conditional_sampling",
        "gradient_2d": "naviertwin.core.flow_analysis.grid_derivatives",
        "find_critical_points": "naviertwin.core.flow_analysis.critical_points",
        "anisotropy_tensor": "naviertwin.core.flow_analysis.anisotropy",
        "connected_components_2d": "naviertwin.core.flow_analysis.morphology",
        "tet_volume": "naviertwin.core.flow_analysis.cell_volume",
    }

    assert len(flow_analysis.__all__) == len(set(flow_analysis.__all__))
    assert set(expected).issubset(set(flow_analysis.__all__))
    for symbol, module_name in expected.items():
        source_module = import_module(module_name)
        assert getattr(flow_analysis, symbol) is getattr(source_module, symbol)


def test_flow_analysis_root_does_not_eagerly_import_optional_modal_backends() -> None:
    """Root import should not require optional modal-analysis backends."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import sys; "
        "import naviertwin.core.flow_analysis; "
        "blocked = {'pydmd', 'pykoopman', 'pysindy', 'pyspod'} & set(sys.modules); "
        "raise SystemExit(1 if blocked else 0)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert completed.returncode == 0


def test_flow_analysis_root_keeps_optional_modal_helpers_explicit() -> None:
    """Optional modal wrappers stay under naviertwin.core.flow_analysis.modal."""
    import naviertwin.core.flow_analysis as flow_analysis

    for symbol in [
        "KoopmanAnalysis",
        "SINDy",
        "compute_spod_pyspod",
        "dmdc_analysis",
        "havok_analysis",
        "hodmd_analysis",
        "mrdmd_analysis",
        "optdmd_analysis",
    ]:
        assert symbol not in flow_analysis.__all__
