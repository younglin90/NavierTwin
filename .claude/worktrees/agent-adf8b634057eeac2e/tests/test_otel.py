"""Round 357 — OpenTelemetry."""

from __future__ import annotations


class TestOtel:
    def test_noop_tracer_works(self) -> None:
        from naviertwin.utils.otel import get_tracer, has_otel

        assert isinstance(has_otel(), bool)
        tracer = get_tracer("test")
        with tracer.span("compute"):
            x = 42
        assert x == 42

    def test_nested_spans(self) -> None:
        from naviertwin.utils.otel import get_tracer

        tracer = get_tracer()
        with tracer.span("outer"):
            with tracer.span("inner"):
                pass
