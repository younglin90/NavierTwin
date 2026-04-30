"""NavierTwin export public API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "NTwinReader": "naviertwin.core.export.ntwin_format",
    "NTwinWriter": "naviertwin.core.export.ntwin_format",
    "load_dataset": "naviertwin.core.export.ntwin_format",
    "save_dataset": "naviertwin.core.export.ntwin_format",
    "export_to_onnx": "naviertwin.core.export.onnx_export",
    "verify_onnx": "naviertwin.core.export.onnx_export",
    "export_to_torchscript": "naviertwin.core.export.torchscript_export",
    "export_to_fmu": "naviertwin.core.export.fmu_export",
    "inspect_fmu": "naviertwin.core.export.fmu_export",
    "validate_fmu_archive": "naviertwin.core.export.fmu_export",
    "trace_and_save": "naviertwin.core.export.torchscript_verify",
    "verify_script_matches": "naviertwin.core.export.torchscript_verify",
    "dynamic_quantize": "naviertwin.core.export.quantize",
    "compare_inference": "naviertwin.core.export.quantize",
    "model_size_bytes": "naviertwin.core.export.quantize",
    "suggest_mlmodel_name": "naviertwin.core.export.coreml_stub",
    "suggest_tflite_name": "naviertwin.core.export.tflite_stub",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Lazily expose export helpers without importing backend stacks eagerly."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable public members for autocomplete and Sphinx."""
    return sorted([*globals(), *__all__])
