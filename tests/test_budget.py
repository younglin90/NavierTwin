"""Round 526 — budget tracker."""

from __future__ import annotations


class TestBudget:
    def test_step_limit(self) -> None:
        from naviertwin.utils.workflow.budget import BudgetTracker

        b = BudgetTracker(max_steps=3)
        for _ in range(3):
            b.tick()
        assert b.exceeded()

    def test_no_limit(self) -> None:
        from naviertwin.utils.workflow.budget import BudgetTracker

        b = BudgetTracker()
        b.tick()
        assert not b.exceeded()
