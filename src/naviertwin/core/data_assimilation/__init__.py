"""Data-assimilation algorithms supporting digital-twin state updates."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "EnKF": "naviertwin.core.data_assimilation.enkf",
    "EnKFSimple": "naviertwin.core.data_assimilation.enkf_simple",
    "ParticleFilter": "naviertwin.core.data_assimilation.particle_filter",
    "KalmanFilter": "naviertwin.core.data_assimilation.kalman",
    "RLS": "naviertwin.core.data_assimilation.rls",
    "ekf_step": "naviertwin.core.data_assimilation.ekf",
    "envar_analysis": "naviertwin.core.data_assimilation.envar",
    "fd_gradient": "naviertwin.core.data_assimilation.var4d_cost",
    "four_dvar_linear": "naviertwin.core.data_assimilation.four_dvar",
    "iekf_step": "naviertwin.core.data_assimilation.iterated_ekf",
    "mhe_estimate": "naviertwin.core.data_assimilation.mhe",
    "nonlinear_rts": "naviertwin.core.data_assimilation.ks_nonlinear",
    "run_filter": "naviertwin.core.data_assimilation.kalman",
    "smooth_particles": "naviertwin.core.data_assimilation.particle_smoother",
    "ukf_step": "naviertwin.core.data_assimilation.ukf",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Lazily expose algorithms without importing every numeric helper eagerly."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable public members used by autocomplete and Sphinx."""
    return sorted([*globals(), *__all__])
