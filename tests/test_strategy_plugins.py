"""Strategy plugin registry contract tests."""

from __future__ import annotations

import pytest

from naviertwin.core.digital_twin.strategies import DataProfile, strategy_report
from naviertwin.core.digital_twin.strategy_plugins import (
    CapabilityAxes,
    RegisteredStrategy,
    StrategyRegistry,
    default_strategy_registry,
)


def _profile(*, dims: int = 2, steps: int = 1, cases: int = 3) -> DataProfile:
    return DataProfile(
        n_cases=cases,
        n_time_steps=steps,
        total_snapshots=cases * steps,
        identical_mesh=True,
        uniform_grid=True,
        n_points=100,
        dims=dims,
        n_params=1,
        topological_dim=dims,
        embedding_dim=3,
    )


def test_default_registry_covers_current_public_methods() -> None:
    registry = default_strategy_registry()

    assert set(registry.keys()) == {
        "rom",
        "physics",
        "dynamics",
        "operator",
        "mesh_gnn",
        "gino",
        "mesh_gnn_mp",
        "transolver",
        "deeponet",
        "mesh_gnn_rollout",
    }
    assert registry.get("operator").capability.spatial_dims == (1, 2, 3)
    assert registry.get("gino").capability.supports_varying_geometry


def test_public_strategy_report_is_backed_by_plugin_registry() -> None:
    profile = _profile()

    assert strategy_report(profile) == default_strategy_registry().report(profile)


def test_geometry_fno_accepts_3d_and_mesh_gnn_rejects_1d() -> None:
    report_3d = default_strategy_registry().report(_profile(dims=3))
    report_1d = default_strategy_registry().report(_profile(dims=1))

    assert report_3d["operator"]["ok"]
    assert report_1d["operator"]["ok"]
    assert not report_1d["mesh_gnn"]["ok"]
    assert "(2, 3)D" in report_1d["mesh_gnn"]["reason"]


def test_registry_rejects_duplicate_keys() -> None:
    capability = CapabilityAxes(
        spatial_dims=(1, 2, 3),
        supports_steady=True,
        supports_unsteady=True,
        supports_case_sets=True,
        supports_varying_geometry=True,
        supports_unstructured_mesh=True,
        requires_uniform_grid=False,
        preprocessing="none",
        compute_backend="test",
    )
    strategy = RegisteredStrategy(
        key="test",
        name="test",
        tier="experimental",
        tier_label="실험적",
        capability=capability,
        checker=lambda _profile: (True, "ok"),
    )
    registry = StrategyRegistry()
    registry.register(strategy)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(strategy)


def test_unknown_strategy_has_clear_error() -> None:
    with pytest.raises(KeyError, match="unknown strategy"):
        StrategyRegistry().get("missing")
