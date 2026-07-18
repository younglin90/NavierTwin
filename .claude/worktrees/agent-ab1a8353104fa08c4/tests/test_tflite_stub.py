"""Round 534 — TFLite stub."""

from __future__ import annotations


class TestTFLite:
    def test_name(self) -> None:
        from naviertwin.core.export.tflite_stub import suggest_tflite_name

        assert suggest_tflite_name("m") == "m.tflite"

    def test_marker(self, tmp_path) -> None:
        from naviertwin.core.export.tflite_stub import write_stub_marker

        p = tmp_path / "s.txt"
        write_stub_marker(p, model_name="x")
        assert "TFLite" in p.read_text()
