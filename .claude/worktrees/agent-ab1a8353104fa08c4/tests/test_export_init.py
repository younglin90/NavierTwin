"""Export package root public API tests."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_export_root_exports_customer_artifact_helpers() -> None:
    """Package root should expose implemented customer export helpers."""
    import naviertwin.core.export as export

    expected = {
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

    assert set(expected).issubset(set(export.__all__))
    for symbol, module_name in expected.items():
        source_module = import_module(module_name)
        assert getattr(export, symbol) is getattr(source_module, symbol)


def test_export_root_does_not_eagerly_import_backend_stacks() -> None:
    """Root import should not load optional ML export backends."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import sys; "
        "import naviertwin.core.export; "
        "blocked = {'torch', 'onnx'} & set(sys.modules); "
        "raise SystemExit(1 if blocked else 0)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert completed.returncode == 0


def test_export_root_keeps_ambiguous_helpers_submodule_explicit() -> None:
    """Collision-prone helper names should stay behind explicit modules."""
    import naviertwin.core.export as export

    for symbol in [
        "has_torch",
        "safe_export",
        "trace_module",
        "write_stub_marker",
    ]:
        assert symbol not in export.__all__
