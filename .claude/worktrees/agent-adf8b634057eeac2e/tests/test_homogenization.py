"""Round 461 — homogenization 1D."""

from __future__ import annotations

import numpy as np


class TestHomog:
    def test_constant(self) -> None:
        from naviertwin.core.multiscale.homogenization import effective_conductivity_1d

        assert effective_conductivity_1d(np.full(10, 2.5)) == 2.5

    def test_two_phase(self) -> None:
        from naviertwin.core.multiscale.homogenization import effective_conductivity_1d

        # 50/50 mix of 1 and 4 → harmonic mean = 1.6
        a = np.array([1.0, 4.0])
        assert abs(effective_conductivity_1d(a) - 1.6) < 1e-12

    def test_cell_solution_periodic(self) -> None:
        from naviertwin.core.multiscale.homogenization import cell_problem_solution

        chi = cell_problem_solution(np.array([1.0, 4.0, 1.0, 4.0]))
        assert abs(chi.mean()) < 1e-12
