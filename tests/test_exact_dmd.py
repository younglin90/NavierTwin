"""Round 216 — exact DMD."""

from __future__ import annotations

import numpy as np


class TestDMD:
    def test_recover_eigenvalues(self) -> None:
        from naviertwin.core.system_id.exact_dmd import exact_dmd

        A = np.array([[0.95, 0.05], [-0.03, 0.9]])
        x = np.array([1.0, 0.5])
        traj = [x]
        for _ in range(30):
            traj.append(A @ traj[-1])
        X = np.array(traj).T
        res = exact_dmd(X, r=2)
        evs = np.sort(res["eigenvalues"].real)
        true_evs = np.sort(np.linalg.eigvals(A).real)
        assert np.allclose(evs, true_evs, atol=1e-6)

    def test_reconstruct_matches(self) -> None:
        from naviertwin.core.system_id.exact_dmd import dmd_reconstruct, exact_dmd

        A = np.array([[0.9, 0.0], [0.0, 0.7]])
        x = np.array([1.0, 1.0])
        traj = [x]
        for _ in range(20):
            traj.append(A @ traj[-1])
        X = np.array(traj).T
        res = exact_dmd(X, r=2)
        t = np.arange(X.shape[1])
        X_re = dmd_reconstruct(res, t)
        assert np.allclose(X, X_re, atol=1e-6)
