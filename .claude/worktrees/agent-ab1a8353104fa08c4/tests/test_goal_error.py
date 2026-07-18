"""Round 375 — goal-oriented error."""

from __future__ import annotations

import numpy as np


class TestGoalError:
    def test_formula(self) -> None:
        from naviertwin.core.amr.goal_error import goal_error, total_error

        R = np.array([1.0, -2.0])
        z = np.array([3.0, 0.5])
        assert np.allclose(goal_error(R, z), [3.0, 1.0])
        assert total_error(R, z) == 4.0
