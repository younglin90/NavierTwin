"""Round 141 — timing utils."""

from __future__ import annotations

import time


class TestTiming:
    def test_decorator(self) -> None:
        from naviertwin.utils.timing import timer

        @timer
        def f(x):
            return x * 2

        assert f(3) == 6

    def test_context(self) -> None:
        from naviertwin.utils.timing import timed

        with timed("block"):
            _ = sum(range(100))

    def test_stopwatch(self) -> None:
        from naviertwin.utils.timing import Stopwatch

        sw = Stopwatch()
        sw.start()
        time.sleep(0.01)
        sw.stop()
        e1 = sw.elapsed
        assert e1 >= 0.005

        # 누적
        sw.start()
        time.sleep(0.01)
        sw.stop()
        assert sw.elapsed >= e1

        sw.reset()
        assert sw.elapsed == 0.0

    def test_context_manager(self) -> None:
        from naviertwin.utils.timing import Stopwatch

        sw = Stopwatch()
        with sw:
            time.sleep(0.005)
        assert sw.elapsed >= 0.002
