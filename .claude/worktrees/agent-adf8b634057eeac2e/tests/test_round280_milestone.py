"""Round 280 — B category milestone: advanced ROM imports + DEIM-LSPG e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneB:
    def test_imports(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear import (  # noqa: F401
            deim,
            gnat,
            local_rom,
            lspg,
            quadratic_manifold,
            shifted_pod,
            space_time_pod,
        )
        from naviertwin.core.system_id import mrdmd, randomized_dmd  # noqa: F401

    def test_deim_lspg_e2e(self) -> None:
        """LSPG + DEIM hyperreduction on a small linear system."""
        from naviertwin.core.dimensionality_reduction.linear.deim import (
            deim,
            deim_project,
        )
        from naviertwin.core.dimensionality_reduction.linear.lspg import lspg_solve

        rng = np.random.default_rng(0)
        n = 30
        A = rng.standard_normal((n, n)) + 5 * np.eye(n)
        b = rng.standard_normal(n)
        # solution-aware basis
        x_true = np.linalg.solve(A, b)
        Q, _ = np.linalg.qr(np.column_stack([x_true, rng.standard_normal((n, 3))]))
        # LSPG reconstruction
        x = lspg_solve(A, b, Q)
        assert np.linalg.norm(x - x_true) < 1e-6
        # DEIM on b basis
        Ub, _, _ = np.linalg.svd(b[:, None] + rng.standard_normal((n, 5)) * 0.01)
        P, idx = deim(Ub[:, :5])
        b_rec = deim_project(Ub[:, :5], P, b[idx])
        assert b_rec.shape == (n,)

    def test_rdmd_modes_e2e(self) -> None:
        from naviertwin.core.system_id.randomized_dmd import randomized_dmd

        rng = np.random.default_rng(2)
        # synthesize DMD-like data
        n = 30
        omega = 0.5
        x0 = rng.standard_normal(n)
        snaps = [x0]
        for k in range(20):
            snaps.append(np.cos(omega * (k + 1)) * x0)
        X = np.column_stack(snaps)
        evals, modes = randomized_dmd(X, rank=3)
        assert modes.shape[0] == n
        assert evals.size == 3
