"""L4 — Operational monitoring loop: drift detect → trigger → lifecycle hop.

Closes the loop using existing R541-R549 utilities. No new code; this is a
contract test that the published API for twin operations actually composes.
"""

from __future__ import annotations

import numpy as np

from naviertwin.core.twin.drift_monitor import psi
from naviertwin.core.twin.lifecycle import TwinFSM
from naviertwin.core.twin.replay_buffer import ReplayBuffer
from naviertwin.core.twin.retrain_trigger import RetrainTrigger
from naviertwin.core.twin.slo import burn_rate


class TestDriftLoop:
    def test_full_loop_reaches_redeploy(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(0.0, 1.0, 1000)

        fsm = TwinFSM()
        fsm.transition("train")
        fsm.transition("deploy")

        buf = ReplayBuffer(capacity=1000)
        trigger = RetrainTrigger(psi_threshold=0.2, cooldown=10)

        # simulate 20 inference batches; mid-stream the data drifts
        retrain_count = 0
        for step in range(20):
            mu = 0.0 if step < 10 else 2.5
            batch = rng.normal(mu, 1.0, 200)
            for v in batch:
                buf.add(float(step), float(v))
            recent = np.asarray([s for _, s in buf.latest(200)])
            score = psi(ref, recent)
            if trigger.should_retrain(psi=score, step=step):
                retrain_count += 1
                fsm.transition("drift")
                fsm.transition("retrain")
                fsm.transition("deploy")

        assert retrain_count >= 1
        assert fsm.state == "deployed"

    def test_slo_burn_signals_alert(self) -> None:
        # 5% errors with 99% SLO → burn rate 5.0 (>2 → alert)
        br = burn_rate(error_count=50, total_count=1000, slo=0.99)
        assert br > 2.0
