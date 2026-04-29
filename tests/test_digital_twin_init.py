"""Digital-twin package root public API tests."""

from __future__ import annotations

from importlib import import_module


def test_digital_twin_root_exports_customer_workflows() -> None:
    """Package root should expose implemented workflow helpers from docs."""
    import naviertwin.core.digital_twin as digital_twin

    expected = {
        "TwinEngine": "naviertwin.core.digital_twin.twin_engine",
        "NavierTwinPipeline": "naviertwin.core.digital_twin.pipeline",
        "PipelineState": "naviertwin.core.digital_twin.pipeline",
        "StreamingDigitalTwin": "naviertwin.core.digital_twin.streaming_twin",
        "batch_predict_fields": "naviertwin.core.digital_twin.batch_predict",
        "build_manifest": "naviertwin.core.digital_twin.manifest",
        "save_manifest": "naviertwin.core.digital_twin.manifest",
        "build_pipeline": "naviertwin.core.digital_twin.pipeline_builder",
        "validate_config": "naviertwin.core.digital_twin.pipeline_builder",
        "compare_models": "naviertwin.core.digital_twin.pipeline_compare",
        "rank_table": "naviertwin.core.digital_twin.pipeline_compare",
    }

    assert set(expected).issubset(set(digital_twin.__all__))
    for symbol, module_name in expected.items():
        source_module = import_module(module_name)
        assert getattr(digital_twin, symbol) is getattr(source_module, symbol)


def test_digital_twin_root_keeps_backend_heavy_extensions_explicit() -> None:
    """Backend-heavy helpers stay in submodules until separately hardened."""
    import naviertwin.core.digital_twin as digital_twin

    for symbol in [
        "auto_tune_pipeline",
        "GNNPipeline",
        "NeuralOperatorPipeline",
        "NeuralState",
        "PINNPipeline",
        "load_pipeline_state",
        "restore_pipeline",
        "save_pipeline_state",
    ]:
        assert symbol not in digital_twin.__all__
