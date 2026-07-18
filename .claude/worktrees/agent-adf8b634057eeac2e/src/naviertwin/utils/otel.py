"""OpenTelemetry tracing — optional `opentelemetry-api` dep.

Falls back to no-op tracer if not installed.

Examples:
    >>> from naviertwin.utils.otel import get_tracer
    >>> tracer = get_tracer("test")
    >>> with tracer.span("op"):
    ...     pass
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


def has_otel() -> bool:
    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        return False
    return True


class _NoopTracer:
    @contextmanager
    def span(self, name: str) -> Any:
        yield None


def get_tracer(name: str = "naviertwin") -> Any:
    if not has_otel():
        return _NoopTracer()
    from opentelemetry import trace
    return _OtelAdapter(trace.get_tracer(name))


class _OtelAdapter:
    def __init__(self, otel_tracer: Any) -> None:
        self._t = otel_tracer

    @contextmanager
    def span(self, name: str) -> Any:
        with self._t.start_as_current_span(name) as s:
            yield s


__all__ = ["get_tracer", "has_otel"]
