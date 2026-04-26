"""V&V — lid-driven cavity Re=100 vs Ghia 1982 centerline u-profile.

Uses ``naviertwin.core.solvers.ns_projection_2d.solve_cavity`` and judges
agreement via ``vv20.is_validated``. Compact mesh + few steps so it stays
in CI budget; tolerance is generous (this is a smoke-grade projection
solver, not a research code).
"""

from __future__ import annotations

import numpy as np
import pytest

from naviertwin.core.benchmarks.ghia_cavity import ghia_u_centerline
from naviertwin.core.verification.vv20 import (
    comparison_error,
    is_validated,
    validation_uncertainty,
)

pytestmark = pytest.mark.vv


class TestCavityGhia:
    def test_centerline_u_profile_re100(self) -> None:
        from naviertwin.core.solvers.ns_projection_2d import solve_cavity

        nx = ny = 33
        u, _, _ = solve_cavity(
            nx=nx, ny=ny, Re=100.0, n_steps=400,
            U_lid=1.0,
        )
        # interpolate solver u at x=0.5 (centerline) onto Ghia y-stations
        y = np.linspace(0, 1, ny)
        ix = nx // 2
        u_center = u[ix, :]  # u(x=0.5, y)
        y_g, u_g = ghia_u_centerline(Re=100)
        u_sim = np.interp(y_g, y, u_center)

        # L∞ error
        err = float(np.max(np.abs(u_sim - u_g)))

        # combined uncertainty: numerical (coarse mesh, few steps) + data
        u_num = 0.30      # coarse projection ~ 0.30 in L∞ tolerance
        u_data = 0.01     # Ghia tabulation precision
        u_val = validation_uncertainty(u_num=u_num, u_input=0.0, u_data=u_data)

        E = comparison_error(S=err, D=0.0)
        assert is_validated(E=E, u_val=u_val, k=2.0), (
            f"max|u_sim - u_ghia| = {err:.3f}, u_val = {u_val:.3f}"
        )
