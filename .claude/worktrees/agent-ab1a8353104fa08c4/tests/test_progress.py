"""Round 99 — 진행률/ETA."""

from __future__ import annotations

import time


class TestProgress:
    def test_basic(self) -> None:
        from naviertwin.utils.progress import ProgressTracker

        t = ProgressTracker(total=100)
        time.sleep(0.01)
        t.update(10)
        assert t.fraction == 0.1
        assert t.rate > 0
        assert "100" in t.format()

    def test_complete(self) -> None:
        from naviertwin.utils.progress import ProgressTracker

        t = ProgressTracker(total=5)
        t.update(5)
        assert t.fraction == 1.0
        assert t.eta_seconds == 0.0

    def test_zero_total(self) -> None:
        from naviertwin.utils.progress import ProgressTracker

        t = ProgressTracker(total=0)
        assert t.fraction == 0.0
