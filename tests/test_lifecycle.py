"""Round 549 — twin lifecycle."""

from __future__ import annotations

import pytest


class TestLifecycle:
    def test_full_path(self) -> None:
        from naviertwin.core.twin.lifecycle import TwinFSM

        f = TwinFSM()
        f.transition("train")
        f.transition("deploy")
        f.transition("drift")
        f.transition("retrain")
        f.transition("deploy")
        assert f.state == "deployed"

    def test_invalid(self) -> None:
        from naviertwin.core.twin.lifecycle import TwinFSM

        f = TwinFSM()
        with pytest.raises(ValueError):
            f.transition("deploy")
