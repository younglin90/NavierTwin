"""Round 246 — watchdog."""

from __future__ import annotations

import time

import pytest


class TestWD:
    def test_ok(self) -> None:
        from naviertwin.utils.watchdog import run_with_timeout

        def f(x):
            return x * 2

        assert run_with_timeout(f, 1.0, 3) == 6

    def test_timeout(self) -> None:
        from naviertwin.utils.watchdog import TimeoutError_, run_with_timeout

        def slow():
            time.sleep(0.5)
            return "done"

        with pytest.raises(TimeoutError_):
            run_with_timeout(slow, 0.05)

    def test_propagates_error(self) -> None:
        from naviertwin.utils.watchdog import run_with_timeout

        def bad():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            run_with_timeout(bad, 1.0)
