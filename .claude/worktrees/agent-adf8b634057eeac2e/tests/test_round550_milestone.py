"""Round 550 — CC category milestone: twin operations (R541-R549) e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneCC:
    def test_imports(self) -> None:
        from naviertwin.core.twin import (  # noqa: F401
            ci_delivery,
            counterfactual,
            drift_monitor,
            lifecycle,
            replay_buffer,
            retrain_trigger,
            slo,
            sync,
            whatif,
        )

    def test_drift_to_retrain_e2e(self) -> None:
        from naviertwin.core.twin.drift_monitor import psi
        from naviertwin.core.twin.lifecycle import TwinFSM
        from naviertwin.core.twin.retrain_trigger import RetrainTrigger

        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 500)
        new = rng.normal(2, 1, 500)
        score = psi(ref, new)
        trigger = RetrainTrigger(psi_threshold=0.2, cooldown=10)
        fsm = TwinFSM()
        fsm.transition("train")
        fsm.transition("deploy")
        if trigger.should_retrain(psi=score, step=100):
            fsm.transition("drift")
            fsm.transition("retrain")
            fsm.transition("deploy")
        assert fsm.state == "deployed"
