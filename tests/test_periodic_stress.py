"""Round 469 — periodic stress homogenization."""

from __future__ import annotations

import numpy as np


class TestPeriodicStress:
    def test_uniform_stress(self) -> None:
        from naviertwin.core.multiscale.periodic_stress import volume_average_stress

        s_one = np.eye(3)
        avg = volume_average_stress(np.tile(s_one, (5, 1, 1)), np.ones(5))
        assert np.allclose(avg, s_one)

    def test_hill_mandel(self) -> None:
        from naviertwin.core.multiscale.periodic_stress import hill_mandel_check

        sigma = np.eye(3)
        eps = 0.5 * np.eye(3)
        # ⟨σ⟩:⟨ε⟩ = 1.5
        assert hill_mandel_check(sigma, eps, micro_work=1.5)
        assert not hill_mandel_check(sigma, eps, micro_work=10.0)
