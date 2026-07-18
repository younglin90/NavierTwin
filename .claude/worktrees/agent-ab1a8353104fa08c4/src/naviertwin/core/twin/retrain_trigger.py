"""Online retraining trigger — based on drift PSI threshold + cooldown.

Examples:
    >>> from naviertwin.core.twin.retrain_trigger import RetrainTrigger
    >>> t = RetrainTrigger(psi_threshold=0.2, cooldown=10)
    >>> t.should_retrain(psi=0.3, step=5)
    True
"""

from __future__ import annotations


class RetrainTrigger:
    def __init__(self, *, psi_threshold: float = 0.2, cooldown: int = 100) -> None:
        self.psi_threshold = psi_threshold
        self.cooldown = cooldown
        self.last_step = -10 ** 9

    def should_retrain(self, *, psi: float, step: int) -> bool:
        if step - self.last_step < self.cooldown:
            return False
        if psi > self.psi_threshold:
            self.last_step = step
            return True
        return False


__all__ = ["RetrainTrigger"]
