"""Round 358 — Prometheus exporter."""

from __future__ import annotations


class TestProm:
    def test_counter(self) -> None:
        from naviertwin.utils.prom_exporter import MetricsRegistry

        reg = MetricsRegistry()
        c = reg.counter("hits")
        c.inc()
        c.inc(4)
        text = reg.format()
        assert "TYPE hits counter" in text
        assert "hits 5" in text

    def test_gauge_and_format(self) -> None:
        from naviertwin.utils.prom_exporter import MetricsRegistry

        reg = MetricsRegistry()
        reg.gauge("queue").set(7.5)
        text = reg.format()
        assert "queue 7.5" in text
