"""검증 지표(Validation Metrics) 모듈.

공개 API:
    - :func:`rmse`: Root Mean Squared Error
    - :func:`r2_score`: 결정계수 R²
    - :func:`relative_l2_error`: 상대 L2 노름 오차
    - :func:`max_error`: 최대 절대 오차
    - :func:`compute_all_metrics`: 모든 지표 일괄 계산
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
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


def __getattr__(name: str) -> Any:
    """Lazily expose validation helpers without importing numeric stacks eagerly."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable public members used by autocomplete and Sphinx."""
    return sorted([*globals(), *__all__])


__all__ = list(_EXPORT_MODULES)
