"""Round 533 — CoreML stub."""

from __future__ import annotations


class TestCoreML:
    def test_name(self) -> None:
        from naviertwin.core.export.coreml_stub import suggest_mlmodel_name

        assert suggest_mlmodel_name("foo") == "foo.mlmodel"

    def test_marker(self, tmp_path) -> None:
        from naviertwin.core.export.coreml_stub import write_stub_marker

        p = tmp_path / "stub.txt"
        write_stub_marker(p, model_name="m")
        assert "CoreML" in p.read_text()
