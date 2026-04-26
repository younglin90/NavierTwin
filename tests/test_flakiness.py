"""Round 568 — flakiness."""

from __future__ import annotations


class TestFlaky:
    def test_stable(self) -> None:
        from naviertwin.utils.flakiness import FlakinessTracker

        t = FlakinessTracker()
        for _ in range(10):
            t.record("solid", True)
        assert t.flakiness("solid") == 0.0

    def test_alternating(self) -> None:
        from naviertwin.utils.flakiness import FlakinessTracker

        t = FlakinessTracker()
        for r in [True, False, True, False, True]:
            t.record("flaky", r)
        # all 4 transitions flip → 1.0
        assert t.flakiness("flaky") == 1.0

    def test_filter(self) -> None:
        from naviertwin.utils.flakiness import FlakinessTracker

        t = FlakinessTracker()
        for r in [True, False, True, True]:
            t.record("a", r)
        for r in [True, True, True, True]:
            t.record("b", r)
        flaky = t.flaky_tests(min_flakiness=0.2)
        assert flaky == ["a"]
