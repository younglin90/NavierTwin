"""Validation package root public API tests."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_validation_root_exports_pure_customer_metrics() -> None:
    """Package root should expose pure validation helpers from docs."""
    import naviertwin.core.validation as validation

    expected = {
        "rmse": "naviertwin.core.validation.metrics",
        "r2_score": "naviertwin.core.validation.metrics",
        "relative_l2_error": "naviertwin.core.validation.metrics",
        "max_error": "naviertwin.core.validation.metrics",
        "compute_all_metrics": "naviertwin.core.validation.metrics",
        "AnalyticSolution": "naviertwin.core.validation.analytic_solutions",
        "couette_flow": "naviertwin.core.validation.analytic_solutions",
        "poiseuille_flow_2d": "naviertwin.core.validation.analytic_solutions",
        "poiseuille_pipe": "naviertwin.core.validation.analytic_solutions",
        "spectral_poiseuille": "naviertwin.core.validation.analytic_solutions",
        "compare_against_analytic": "naviertwin.core.validation.analytic_solutions",
        "kfold_scores": "naviertwin.core.validation.cross_val",
        "grid_search": "naviertwin.core.validation.cross_val",
        "wasserstein_1d": "naviertwin.core.validation.distances",
        "mmd_gaussian": "naviertwin.core.validation.distances",
        "kl_divergence_gaussian": "naviertwin.core.validation.distances",
        "field_diff_stats": "naviertwin.core.validation.field_diff",
        "hotspot_indices": "naviertwin.core.validation.field_diff",
        "band_mask": "naviertwin.core.validation.field_diff",
        "field_sanity_check": "naviertwin.core.validation.field_sanity",
        "detect_outliers_iqr": "naviertwin.core.validation.field_sanity",
        "detect_outliers_zscore": "naviertwin.core.validation.field_sanity",
        "psnr": "naviertwin.core.validation.image_metrics",
        "nrmse": "naviertwin.core.validation.image_metrics",
        "ssim": "naviertwin.core.validation.image_metrics",
        "channel_rmse": "naviertwin.core.validation.multi_output_metrics",
        "channel_relative_error": "naviertwin.core.validation.multi_output_metrics",
        "aggregated_rmse": "naviertwin.core.validation.multi_output_metrics",
        "multi_output_r2": "naviertwin.core.validation.multi_output_metrics",
        "cross_channel_correlation": "naviertwin.core.validation.multi_output_metrics",
        "top_k_worst_channels": "naviertwin.core.validation.multi_output_metrics",
        "per_sample_error_norm": "naviertwin.core.validation.multi_output_metrics",
        "taylor_green_2d": "naviertwin.core.validation.taylor_green",
        "kinetic_energy_decay": "naviertwin.core.validation.taylor_green",
    }

    assert set(expected).issubset(set(validation.__all__))
    for symbol, module_name in expected.items():
        source_module = import_module(module_name)
        assert getattr(validation, symbol) is getattr(source_module, symbol)


def test_validation_root_does_not_eagerly_import_preflight_reader_stack() -> None:
    """Pure root import should not pull in CFD-reader preflight dependencies."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import sys; "
        "import naviertwin.core.validation; "
        "blocked = [m for m in sys.modules "
        "if m == 'naviertwin.core.validation.dataset_preflight' "
        "or m.startswith('naviertwin.core.cfd_reader')]; "
        "raise SystemExit(1 if blocked else 0)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert completed.returncode == 0


def test_validation_root_keeps_preflight_module_explicit() -> None:
    """Preflight stays module-qualified because it reaches the reader stack."""
    import naviertwin.core.validation as validation

    for symbol in [
        "build_dataset_preflight_report",
        "format_preflight_report",
        "report_to_json",
    ]:
        assert symbol not in validation.__all__
