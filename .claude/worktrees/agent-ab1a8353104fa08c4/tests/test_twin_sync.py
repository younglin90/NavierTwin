"""Round 541 — twin sync."""

from __future__ import annotations

import numpy as np


class TestSync:
    def test_blend(self) -> None:
        from naviertwin.core.twin.sync import sync_state

        out = sync_state(np.zeros(3), np.ones(3), alpha=0.5)
        assert np.allclose(out, 0.5)

    def test_no_drift(self) -> None:
        from naviertwin.core.twin.sync import sync_state

        out = sync_state(np.ones(3), np.ones(3), alpha=1.0)
        assert np.allclose(out, 1.0)
