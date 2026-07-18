"""Digital-twin workflow public API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "TwinEngine": "naviertwin.core.digital_twin.twin_engine",
    "NavierTwinPipeline": "naviertwin.core.digital_twin.pipeline",
    "PipelineState": "naviertwin.core.digital_twin.pipeline",
    "PhysicsAITwinEngine": "naviertwin.core.digital_twin.physics_ai_engine",
    "GeometryFNOTwinEngine": "naviertwin.core.digital_twin.geometry_fno_engine",
    "DMDTwinEngine": "naviertwin.core.digital_twin.dmd_engine",
    "StreamingDigitalTwin": "naviertwin.core.digital_twin.streaming_twin",
    "batch_predict_fields": "naviertwin.core.digital_twin.batch_predict",
    "build_manifest": "naviertwin.core.digital_twin.manifest",
    "save_manifest": "naviertwin.core.digital_twin.manifest",
    "build_pipeline": "naviertwin.core.digital_twin.pipeline_builder",
    "validate_config": "naviertwin.core.digital_twin.pipeline_builder",
    "compare_models": "naviertwin.core.digital_twin.pipeline_compare",
    "rank_table": "naviertwin.core.digital_twin.pipeline_compare",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Lazily expose workflow helpers without importing every backend eagerly."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable public members used by autocomplete and Sphinx."""
    return sorted([*globals(), *__all__])
