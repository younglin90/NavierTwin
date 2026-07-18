"""Round 340 — H category milestone: optimization (R331-R339) + Tchebycheff."""

from __future__ import annotations

import numpy as np


class TestMilestoneH:
    def test_imports(self) -> None:
        from naviertwin.core.optimization import (  # noqa: F401
            aug_lagrangian,
            bobyqa,
            mads,
            nelder_mead,
            nsga2_constrained,
            shape_opt,
            sqp,
            tchebycheff,
            topo_simp,
            trust_region,
        )

    def test_tchebycheff(self) -> None:
        from naviertwin.core.optimization.tchebycheff import (
            tchebycheff,
            weight_grid,
        )

        # max(0.5*|1|, 0.5*|2|) = 1.0
        assert tchebycheff(
            np.array([1.0, 2.0]), np.array([0.5, 0.5]), np.zeros(2),
        ) == 1.0
        W = weight_grid(2, 4)
        assert W.shape == (4, 2)
        assert np.allclose(W.sum(axis=1), 1.0)

    def test_multi_obj_pipeline_e2e(self) -> None:
        """Tchebycheff scalarization + Nelder-Mead = single-objective subproblem."""
        from naviertwin.core.optimization.nelder_mead import nelder_mead
        from naviertwin.core.optimization.tchebycheff import tchebycheff

        # objectives: f1=x², f2=(x-2)²; pareto front in [0, 2]
        def scalar(x):
            f = np.array([x[0] ** 2, (x[0] - 2) ** 2])
            return tchebycheff(f, np.array([0.5, 0.5]), np.zeros(2))

        x = nelder_mead(scalar, x0=np.array([1.5]), max_iter=300)
        # Tchebycheff (w=0.5,0.5, z*=0) optimum: |x²|=|x-2|² → x=1
        assert abs(x[0] - 1.0) < 0.05
