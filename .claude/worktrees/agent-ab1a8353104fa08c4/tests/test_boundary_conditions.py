"""Round 127 — 경계조건."""

from __future__ import annotations

import numpy as np
import pytest


class TestBC:
    def test_dirichlet(self) -> None:
        from naviertwin.core.solvers.boundary_conditions import apply_bc, make_bc

        u = np.arange(5, dtype=float)
        apply_bc(u, left=make_bc("dirichlet", 10.0), right=make_bc("dirichlet", -1.0))
        assert u[0] == 10.0
        assert u[-1] == -1.0

    def test_neumann_zero(self) -> None:
        from naviertwin.core.solvers.boundary_conditions import apply_bc, make_bc

        u = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
        apply_bc(u, left=make_bc("neumann", 0.0), right=make_bc("neumann", 0.0), dx=0.1)
        # ∂u/∂n=0 → u_0 = u_1, u_{-1} = u_{-2}
        assert u[0] == u[1]
        assert u[-1] == u[-2]

    def test_periodic(self) -> None:
        from naviertwin.core.solvers.boundary_conditions import apply_bc, make_bc

        u = np.array([0.0, 2.0, 3.0, 4.0, 0.0])
        apply_bc(u, left=make_bc("periodic"), right=make_bc("periodic"))
        assert u[0] == u[-2]
        assert u[-1] == u[1]

    def test_robin(self) -> None:
        from naviertwin.core.solvers.boundary_conditions import apply_bc, make_bc

        u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # α=1, β=0 → dirichlet 등가
        apply_bc(u, left=make_bc("robin", 7.0, alpha=1.0, beta=0.0), dx=1.0)
        assert u[0] == 7.0

    def test_callable_value(self) -> None:
        from naviertwin.core.solvers.boundary_conditions import apply_bc, make_bc

        u = np.zeros(5)
        bc = make_bc("dirichlet", value=lambda t: np.sin(t))
        apply_bc(u, left=bc, t=np.pi / 2)
        assert abs(u[0] - 1.0) < 1e-10

    def test_invalid(self) -> None:
        from naviertwin.core.solvers.boundary_conditions import apply_bc, make_bc

        with pytest.raises(ValueError):
            apply_bc(np.zeros(3), left=make_bc("bogus"))
