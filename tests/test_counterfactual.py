"""Round 548 — counterfactual."""

from __future__ import annotations

import numpy as np


class TestCF:
    def test_linear(self) -> None:
        from naviertwin.core.twin.counterfactual import minimal_change

        # f(x) = x; target=2 from 0 → δ should be positive
        delta = minimal_change(np.array([0.0]),
                                lambda x: float(x[0]),
                                target=2.0, lr=0.1, n_iter=200, lam=0.01)
        assert delta[0] > 0
