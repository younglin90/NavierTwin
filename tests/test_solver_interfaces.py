"""Round 22 — LBM + Lettuce/flowtorch/JAX-Fluids 래퍼."""

from __future__ import annotations

import numpy as np
import pytest


class TestLBMD2Q9:
    def test_run_produces_snapshots(self) -> None:
        from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

        lbm = LBMD2Q9(nx=16, ny=16, tau=0.8, u_top=0.05)
        snaps = lbm.run(n_steps=50, record_every=25)
        assert snaps.shape == (2, 16, 16, 3)
        assert np.all(np.isfinite(snaps))
        # 밀도 양수
        assert np.all(snaps[..., 0] > 0)

    def test_invalid_tau(self) -> None:
        from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

        with pytest.raises(ValueError):
            LBMD2Q9(nx=8, ny=8, tau=0.3)

    def test_cavity_drives_top_velocity(self) -> None:
        """상단 행 ux 가 u_top 과 일치해야 (moving-wall 강제)."""
        from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

        lbm = LBMD2Q9(nx=12, ny=12, tau=0.9, u_top=0.1)
        snaps = lbm.run(n_steps=30, record_every=30)
        # 마지막 스냅의 상단 행 ux
        top_ux = snaps[0, 0, :, 1]
        assert np.allclose(top_ux, 0.1, atol=1e-12)


class TestLettuceWrapper:
    def test_run_cavity(self) -> None:
        from naviertwin.core.solver_interfaces.lettuce_wrapper import run_cavity

        snaps = run_cavity(nx=12, ny=12, n_steps=40, record_every=20)
        assert snaps.shape == (2, 12, 12, 3)


class TestFlowtorchWrapper:
    def test_pod_gpu(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.solver_interfaces.flowtorch_wrapper import pod_gpu

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 15))
        U, s, V = pod_gpu(X, n_modes=5)
        assert U.shape == (30, 5)
        assert s.shape == (5,)
        assert V.shape == (15, 5)
        # 특이값 내림차순
        assert all(s[i] >= s[i + 1] for i in range(len(s) - 1))


class TestJAXFluidsWrapper:
    def test_availability_check(self) -> None:
        from naviertwin.core.solver_interfaces.jax_fluids_wrapper import (
            jax_fluids_available,
            require_jax_fluids,
        )

        avail = jax_fluids_available()
        assert isinstance(avail, bool)
        if not avail:
            with pytest.raises(RuntimeError, match="JAX-Fluids"):
                require_jax_fluids()
