"""Turbulence package root public API tests."""

from __future__ import annotations

from importlib import import_module


def test_turbulence_root_exports_diagnostic_api() -> None:
    """Package root should expose shipped turbulence diagnostics."""
    import naviertwin.core.turbulence as turbulence

    expected = {
        "energy_spectrum_1d": "naviertwin.core.turbulence.energy_spectrum",
        "energy_spectrum_2d": "naviertwin.core.turbulence.energy_spectrum",
        "kolmogorov_slope": "naviertwin.core.turbulence.energy_spectrum",
        "eddy_viscosity": "naviertwin.core.turbulence.k_epsilon",
        "production_rate": "naviertwin.core.turbulence.k_epsilon",
        "k_epsilon_step": "naviertwin.core.turbulence.k_epsilon",
    }

    assert set(expected).issubset(set(turbulence.__all__))
    for symbol, module_name in expected.items():
        source_module = import_module(module_name)
        assert getattr(turbulence, symbol) is getattr(source_module, symbol)
