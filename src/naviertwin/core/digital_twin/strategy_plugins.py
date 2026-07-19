"""Extensible strategy registry and backend protocol.

Capability inspection stays in core. Heavy implementations can be supplied by
optional packages through the ``naviertwin.strategies`` entry-point group.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class CapabilityAxes:
    """Explicit data axes supported by one learning strategy."""

    spatial_dims: tuple[int, ...]
    supports_steady: bool
    supports_unsteady: bool
    supports_case_sets: bool
    supports_varying_geometry: bool
    supports_unstructured_mesh: bool
    requires_uniform_grid: bool
    preprocessing: str
    compute_backend: str

    def __post_init__(self) -> None:
        if not self.spatial_dims or any(value not in (1, 2, 3) for value in self.spatial_dims):
            raise ValueError("spatial_dims must contain values from 1, 2, 3")


@dataclass(frozen=True)
class StrategyDecision:
    """Pre-training eligibility decision rendered directly by the UI."""

    ok: bool
    reason: str
    name: str
    tier: str
    tier_label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "name": self.name,
            "tier": self.tier,
            "tier_label": self.tier_label,
        }


@runtime_checkable
class StrategyBackend(Protocol):
    """Execution contract implemented by built-in or optional model backends."""

    def prepare(self, case_set: Any, config: Mapping[str, Any]) -> Any:
        """Convert canonical cases into backend-native tensors/graphs."""

    def fit(
        self,
        prepared: Any,
        config: Mapping[str, Any],
        progress: Callable[[float, str], None] | None = None,
    ) -> Any:
        """Train and return a serializable model artifact."""

    def predict(self, artifact: Any, query: Mapping[str, Any]) -> Any:
        """Predict one field bundle on requested coordinates/time."""

    def validate(self, artifact: Any, evaluation_data: Any) -> Any:
        """Return strategy-specific and common validation results."""


@dataclass(frozen=True)
class RegisteredStrategy:
    """Metadata, checker, and optional executable backend factory."""

    key: str
    name: str
    tier: str
    tier_label: str
    capability: CapabilityAxes
    checker: Callable[[Any], tuple[bool, str]]
    backend_factory: Callable[[], StrategyBackend] | None = None

    def inspect(self, profile: Any) -> StrategyDecision:
        ok, reason = self.checker(profile)
        return StrategyDecision(
            ok=bool(ok),
            reason=str(reason),
            name=self.name,
            tier=self.tier,
            tier_label=self.tier_label,
        )


class StrategyRegistry:
    """Deterministic registry with duplicate protection and plugin discovery."""

    def __init__(self) -> None:
        self._items: dict[str, RegisteredStrategy] = {}

    def register(self, strategy: RegisteredStrategy) -> None:
        if not strategy.key:
            raise ValueError("strategy key must be non-empty")
        if strategy.key in self._items:
            raise ValueError(f"strategy already registered: {strategy.key}")
        self._items[strategy.key] = strategy

    def get(self, key: str) -> RegisteredStrategy:
        try:
            return self._items[key]
        except KeyError as exc:
            raise KeyError(f"unknown strategy: {key}") from exc

    def keys(self) -> tuple[str, ...]:
        return tuple(self._items)

    def report(self, profile: Any) -> dict[str, dict[str, Any]]:
        return {
            key: strategy.inspect(profile).to_dict()
            for key, strategy in self._items.items()
        }

    def load_entry_points(self, group: str = "naviertwin.strategies") -> list[str]:
        """Load optional strategy factories without making core imports heavy."""

        loaded: list[str] = []
        discovered = metadata.entry_points()
        entries = discovered.select(group=group) if hasattr(discovered, "select") else ()
        for entry in entries:
            factory = entry.load()
            strategy = factory()
            if not isinstance(strategy, RegisteredStrategy):
                raise TypeError(
                    f"strategy entry point {entry.name!r} did not return RegisteredStrategy"
                )
            self.register(strategy)
            loaded.append(strategy.key)
        return loaded


_CAPABILITIES: dict[str, CapabilityAxes] = {
    "rom": CapabilityAxes(
        (1, 2, 3), True, True, True, False, True, False, "identical-or-remap", "numpy",
    ),
    "physics": CapabilityAxes(
        (1, 2, 3), True, True, True, True, True, False, "point-cloud", "torch",
    ),
    "dynamics": CapabilityAxes(
        (1, 2, 3), False, True, True, False, True, False, "identical-mesh", "numpy",
    ),
    "operator": CapabilityAxes(
        (1, 2, 3), True, True, True, True, False, True, "uniform-grid+sdf", "torch",
    ),
    "mesh_gnn": CapabilityAxes(
        (2, 3), True, True, True, True, True, False, "mesh-graph", "torch-geometric",
    ),
    "gino": CapabilityAxes(
        (2, 3), True, True, True, True, True, False, "point-cloud+latent-grid", "neuraloperator",
    ),
    "mesh_gnn_mp": CapabilityAxes(
        (2, 3), True, True, True, True, True, False, "mesh-graph+edge-features", "torch-geometric",
    ),
    "transolver": CapabilityAxes(
        (2, 3), True, True, True, True, True, False, "physics-slices", "torch",
    ),
    # 단일 케이스 시계열 전용 — supports_steady=False(정상 데이터엔 적용 안 됨),
    # supports_case_sets=False(다른 7개와 정반대 방향).
    "mesh_gnn_rollout": CapabilityAxes(
        (2, 3), False, True, False, False, True, False,
        "mesh-graph+trajectory-rollout", "torch-geometric",
    ),
}


def default_strategy_registry() -> StrategyRegistry:
    """Adapt all currently wired strategies to the plugin registry."""

    from naviertwin.core.digital_twin.strategies import STRATEGIES, TIER_LABELS, _check

    registry = StrategyRegistry()
    for spec in STRATEGIES:
        capability = _CAPABILITIES[spec.key]

        def checker(
            profile: Any,
            current: Any = spec,
            current_capability: CapabilityAxes = capability,
        ) -> tuple[bool, str]:
            dimension = int(
                getattr(profile, "topological_dim", getattr(profile, "dims", 3))
            )
            if dimension not in current_capability.spatial_dims:
                return False, f"{current_capability.spatial_dims}D 공간 차원만 지원합니다."
            return _check(current, profile)

        registry.register(
            RegisteredStrategy(
                key=spec.key,
                name=spec.name,
                tier=spec.tier,
                tier_label=TIER_LABELS.get(spec.tier, spec.tier),
                capability=capability,
                checker=checker,
            )
        )
    return registry


__all__ = [
    "CapabilityAxes",
    "RegisteredStrategy",
    "StrategyBackend",
    "StrategyDecision",
    "StrategyRegistry",
    "default_strategy_registry",
]
