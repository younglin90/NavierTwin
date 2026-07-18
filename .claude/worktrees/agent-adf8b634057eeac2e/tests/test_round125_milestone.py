"""Round 125 — 마일스톤: R101-124 모듈 import + Burgers → POD → surrogate e2e."""

from __future__ import annotations

import pytest

R101_124 = [
    "naviertwin.core.validation.taylor_green",
    "naviertwin.core.analysis.streamline",
    "naviertwin.utils.complex_step",
    "naviertwin.core.analysis.vorticity",
    "naviertwin.core.analysis.aero_forces",
    "naviertwin.core.analysis.boundary_layer",
    "naviertwin.core.analysis.rom_energy",
    "naviertwin.core.tools.mesh_stats",
    "naviertwin.core.analysis.interpolate",
    "naviertwin.core.analysis.probe",
    "naviertwin.core.analysis.time_integrator",
    "naviertwin.core.koopman.linear_fit",
    "naviertwin.core.data_assimilation.kalman",
    "naviertwin.core.neural.mlp_blocks",
    "naviertwin.core.neural.pde_residuals",
    "naviertwin.core.linalg.iterative_solvers",
    "naviertwin.core.linalg.svd_utils",
    "naviertwin.core.linalg.sparse_builder",
    "naviertwin.core.solvers.fd_1d",
    "naviertwin.core.solvers.fd_2d",
    "naviertwin.core.linalg.tridiagonal",
    "naviertwin.core.solvers.conv_diff_2d",
    "naviertwin.core.analysis.roi_mask",
    "naviertwin.core.validation.field_sanity",
]


class TestRound125:
    @pytest.mark.parametrize("m", R101_124)
    def test_importable(self, m: str) -> None:
        import importlib
        importlib.import_module(m)

    def test_burgers_to_pod_e2e(self) -> None:
        """1D Burgers FD 시뮬 → POD → 에너지 분석 → 선형 dynamics fit."""
        pytest.importorskip("sklearn")
        from naviertwin.core.analysis.rom_energy import (
            energy_retention,
            n_modes_for_energy,
        )
        from naviertwin.core.koopman.linear_fit import (
            eigenanalysis,
            fit_linear_dynamics,
        )
        from naviertwin.core.linalg.svd_utils import truncated_svd
        from naviertwin.core.solvers.fd_1d import solve_burgers_1d
        from naviertwin.core.validation.field_sanity import field_sanity_check

        # 1) 시뮬레이션
        x, t, U = solve_burgers_1d(nx=64, T=0.3, nu=0.01)
        assert U.shape[1] > 10
        sanity = field_sanity_check(U)
        assert sanity["all_finite"]

        # 2) POD (SVD)
        Ucent = U - U.mean(axis=1, keepdims=True)
        Umod, s, _ = truncated_svd(Ucent, k=10)
        ret = energy_retention(s)
        assert ret[-1] == pytest.approx(1.0)
        n99 = n_modes_for_energy(s, threshold=0.99)
        assert 1 <= n99 <= 10

        # 3) reduced dynamics (coeffs through time)
        coeffs = Umod.T @ Ucent  # (k, T)
        A = fit_linear_dynamics(coeffs)
        info = eigenanalysis(A)
        assert A.shape == (coeffs.shape[0], coeffs.shape[0])
        # Burgers 는 감쇠 → 대부분 eigenvalue |λ|<=1
        assert float(info["magnitudes"].max()) < 1.5
