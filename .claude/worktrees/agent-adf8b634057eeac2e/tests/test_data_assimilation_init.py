"""Data-assimilation package root public API tests."""

from __future__ import annotations

from importlib import import_module


def test_data_assimilation_root_exports_customer_algorithms() -> None:
    """Package root should expose implemented assimilation algorithms from docs."""
    import naviertwin.core.data_assimilation as da

    expected = {
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

    assert set(expected).issubset(set(da.__all__))
    for symbol, module_name in expected.items():
        source_module = import_module(module_name)
        assert getattr(da, symbol) is getattr(source_module, symbol)


def test_data_assimilation_root_preserves_submodule_imports() -> None:
    """Lazy symbol exports should not break legacy submodule imports."""
    from naviertwin.core.data_assimilation import (
        ekf,
        particle_smoother,
        rls,
        rts_smoother,
        var4d_cost,
    )

    assert ekf.__name__.endswith(".ekf")
    assert particle_smoother.__name__.endswith(".particle_smoother")
    assert rls.__name__.endswith(".rls")
    assert rts_smoother.__name__.endswith(".rts_smoother")
    assert var4d_cost.__name__.endswith(".var4d_cost")
