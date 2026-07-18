"""Round 220 — 마일스톤 R211-R219."""

from __future__ import annotations

import numpy as np
import pytest

R211_219 = [
    "naviertwin.core.uncertainty.hmc",
    "naviertwin.core.uncertainty.vi",
    "naviertwin.core.neural.neural_sde",
    "naviertwin.core.dimensionality_reduction.linear.bpod_scratch",
    "naviertwin.core.dimensionality_reduction.linear.weighted_pod",
    "naviertwin.core.system_id.exact_dmd",
    "naviertwin.core.linalg.newton_krylov",
    "naviertwin.core.neural.neural_ode",
    "naviertwin.core.solvers.dg_1d",
]


class TestRound220:
    @pytest.mark.parametrize("m", R211_219)
    def test_importable(self, m: str) -> None:
        import importlib
        importlib.import_module(m)

    def test_dmd_then_bpod(self) -> None:
        """DMD 고유값 체크 후 BPOD 축소 차수 구성."""
        from naviertwin.core.dimensionality_reduction.linear.bpod_scratch import (
            bpod_reduce,
        )
        from naviertwin.core.system_id.exact_dmd import exact_dmd

        A = np.array([[0.9, 0.1], [-0.05, 0.85]])
        x = np.array([1.0, 0.5])
        traj = [x]
        for _ in range(30):
            traj.append(A @ traj[-1])
        X = np.array(traj).T
        res = exact_dmd(X, r=2)
        assert res["eigenvalues"].shape == (2,)

        B = np.array([[1.0], [0.5]])
        C = np.array([[1.0, 1.0]])
        Ar, Br, Cr, _, _ = bpod_reduce(A, B, C, r=2)
        assert Ar.shape == (2, 2)

    def test_hmc_vs_mh(self) -> None:
        from naviertwin.core.uncertainty.hmc import hmc
        from naviertwin.core.uncertainty.mcmc import metropolis_hastings

        def logp(q):
            return -0.5 * float(q[0] ** 2)

        s_mh = metropolis_hastings(logp, np.zeros(1), n=500, step=1.0, seed=0)
        s_hmc = hmc(logp, np.zeros(1), n=500, step=0.1, L=10,
                    grad=lambda q: -q, seed=0)
        # 두 방법 모두 평균 ≈ 0, 분산 근처 1
        assert abs(float(s_mh.mean())) < 0.3
        assert abs(float(s_hmc.mean())) < 0.3
