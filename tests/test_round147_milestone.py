"""Round 147 — 마일스톤: R126-R146 import + 종합 e2e 스모크."""

from __future__ import annotations

import numpy as np
import pytest

R126_146 = [
    "naviertwin.core.dimensionality_reduction.linear.galerkin",
    "naviertwin.core.solvers.boundary_conditions",
    "naviertwin.core.analysis.spectral",
    "naviertwin.core.linalg.nonlinear",
    "naviertwin.core.linalg.least_squares",
    "naviertwin.core.optimization.adjoint",
    "naviertwin.core.optimization.gradient_opt",
    "naviertwin.core.optimization.line_search",
    "naviertwin.core.optimization.bfgs",
    "naviertwin.core.analysis.rbf_interp",
    "naviertwin.core.numerics.clenshaw_curtis",
    "naviertwin.core.report.csv_writer",
    "naviertwin.core.report.markdown",
    "naviertwin.core.report.html_report",
    "naviertwin.utils.experiment_log",
    "naviertwin.utils.timing",
    "naviertwin.core.solvers.pressure_poisson",
    "naviertwin.core.solvers.ns_projection_2d",
    "naviertwin.core.neural.transformer_block",
    "naviertwin.core.neural.conv_ae",
    "naviertwin.core.neural.vae",
]


class TestRound147:
    @pytest.mark.parametrize("m", R126_146)
    def test_importable(self, m: str) -> None:
        import importlib
        importlib.import_module(m)

    def test_end_to_end_cavity_to_rom(self) -> None:
        """NS cavity → snapshots → SVD → Galerkin 선형 ROM."""
        from naviertwin.core.dimensionality_reduction.linear.galerkin import (
            project_field_to_modes,
            reconstruct_from_modes,
        )
        from naviertwin.core.linalg.svd_utils import truncated_svd
        from naviertwin.core.solvers.ns_projection_2d import solve_cavity
        from naviertwin.core.validation.field_sanity import field_sanity_check

        # 작은 cavity 3번 시뮬 (Re 변화)
        snaps = []
        for Re in (10.0, 25.0, 50.0):
            u, v, _ = solve_cavity(nx=12, ny=12, Re=Re, n_steps=20)
            snaps.append(np.stack([u.ravel(), v.ravel()]))  # 각 (2, n_points)
        X = np.stack(snaps, axis=-1).reshape(-1, 3)  # (2*144, 3)
        assert field_sanity_check(X)["all_finite"]

        U, s, _ = truncated_svd(X - X.mean(axis=1, keepdims=True), k=2)
        assert s[0] >= s[1]

        # 계수 투영/재구성
        a = project_field_to_modes(U, X - X.mean(axis=1, keepdims=True))
        X_re = reconstruct_from_modes(U, a) + X.mean(axis=1, keepdims=True)
        assert np.linalg.norm(X - X_re) / np.linalg.norm(X) < 0.3
