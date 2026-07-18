"""Round 544 — retrain trigger."""

from __future__ import annotations


class TestRetrain:
    def test_fire_then_cooldown(self) -> None:
        from naviertwin.core.twin.retrain_trigger import RetrainTrigger

        t = RetrainTrigger(psi_threshold=0.2, cooldown=10)
        assert t.should_retrain(psi=0.3, step=5)
        # next call within cooldown blocked
        assert not t.should_retrain(psi=0.5, step=8)
        # after cooldown
        assert t.should_retrain(psi=0.3, step=20)

    def test_no_fire_below_threshold(self) -> None:
        from naviertwin.core.twin.retrain_trigger import RetrainTrigger

        t = RetrainTrigger(psi_threshold=0.5, cooldown=1)
        assert not t.should_retrain(psi=0.1, step=5)
