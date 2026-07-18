"""Round 545 — CI delivery."""

from __future__ import annotations


class TestCI:
    def test_bounds(self) -> None:
        from naviertwin.core.twin.ci_delivery import wrap_with_ci

        r = wrap_with_ci(value=10.0, sigma=2.0, z=2.0)
        assert r["lower"] == 6.0
        assert r["upper"] == 14.0
        assert r["value"] == 10.0
