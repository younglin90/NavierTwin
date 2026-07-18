"""Round 190 — 마일스톤 R181-R189."""

from __future__ import annotations

import numpy as np
import pytest

R181_189 = [
    "naviertwin.core.uncertainty.mcmc",
    "naviertwin.core.system_id.delay_embed",
    "naviertwin.core.analysis.lyapunov",
    "naviertwin.core.analysis.wavelet",
    "naviertwin.core.analysis.energy_spectrum",
    "naviertwin.core.optimization.inverse_design",
    "naviertwin.core.analysis.barycentric",
    "naviertwin.core.tools.mesh_quality",
    "naviertwin.core.linalg.power_method",
]


class TestRound190:
    @pytest.mark.parametrize("m", R181_189)
    def test_importable(self, m: str) -> None:
        import importlib
        importlib.import_module(m)

    def test_inverse_design_via_power(self) -> None:
        """합성: SPD A 의 dominant 고유값 → 역설계 대상."""
        from naviertwin.core.linalg.power_method import power_iteration
        from naviertwin.core.optimization.inverse_design import inverse_design

        rng = np.random.default_rng(0)
        M = rng.standard_normal((5, 5))
        A = M @ M.T
        lam, _ = power_iteration(A, n_iter=200)

        # inverse design: find scalar p such that p^2 ≈ lam
        def fwd(p):
            return np.array([p[0] ** 2])

        p, _ = inverse_design(fwd, np.array([lam]), np.array([1.0]),
                              lr=0.05, n_iter=500)
        assert abs(p[0] ** 2 - lam) < 1e-3
