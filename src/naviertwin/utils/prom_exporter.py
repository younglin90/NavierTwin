"""Prometheus metrics exporter — text format (no external dep).

Examples:
    >>> from naviertwin.utils.prom_exporter import MetricsRegistry
    >>> reg = MetricsRegistry()
    >>> reg.counter("requests").inc(3)
    >>> "requests 3" in reg.format()
    True
"""

from __future__ import annotations


class _Counter:
    def __init__(self) -> None:
        self.value = 0.0

    def inc(self, n: float = 1.0) -> None:
        self.value += n


class _Gauge:
    def __init__(self) -> None:
        self.value = 0.0

    def set(self, v: float) -> None:
        self.value = v


class MetricsRegistry:
    def __init__(self) -> None:
        self._counters: dict[str, _Counter] = {}
        self._gauges: dict[str, _Gauge] = {}

    def counter(self, name: str) -> _Counter:
        return self._counters.setdefault(name, _Counter())

    def gauge(self, name: str) -> _Gauge:
        return self._gauges.setdefault(name, _Gauge())

    def format(self) -> str:
        lines = []
        counters = list(self._counters.items())
        idx = 0
        while idx < len(counters):
            name, c = counters[idx]
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {c.value}")
            idx += 1
        gauges = list(self._gauges.items())
        idx = 0
        while idx < len(gauges):
            name, g = gauges[idx]
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {g.value}")
            idx += 1
        return "\n".join(lines) + "\n"


__all__ = ["MetricsRegistry"]
