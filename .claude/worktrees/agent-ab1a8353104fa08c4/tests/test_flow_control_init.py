"""Flow-control package root public API tests."""

from __future__ import annotations

from importlib import import_module


def test_flow_control_root_exports_policy_api() -> None:
    """Package root should expose shipped policy optimization helpers."""
    import naviertwin.core.flow_control as flow_control

    expected = {
        "GaussianPolicy": "naviertwin.core.flow_control.policy_gradient",
        "reinforce_update": "naviertwin.core.flow_control.policy_gradient",
    }

    assert set(expected).issubset(set(flow_control.__all__))
    for symbol, module_name in expected.items():
        source_module = import_module(module_name)
        assert getattr(flow_control, symbol) is getattr(source_module, symbol)
