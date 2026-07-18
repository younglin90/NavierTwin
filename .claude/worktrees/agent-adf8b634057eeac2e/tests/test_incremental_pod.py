"""Round 605 — IncrementalPOD streaming update coverage."""

from __future__ import annotations

import numpy as np
import pytest


class TestIncrementalPOD:
    def _make_snapshots(self, n_space: int = 60, n_snap: int = 30, seed: int = 0):
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((n_space, 5))
        V = rng.standard_normal((5, n_snap))
        return U @ V + 0.01 * rng.standard_normal((n_space, n_snap))

    def test_single_update(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        pod = IncrementalPOD(n_modes=5)
        snap = np.random.default_rng(0).standard_normal(50)
        pod.update(snap)
        assert pod.is_fitted
        assert pod.n_snapshots == 1

    def test_multiple_updates_basis_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        X = self._make_snapshots()
        pod = IncrementalPOD(n_modes=5)
        for j in range(X.shape[1]):
            pod.update(X[:, j])
        assert pod.basis is not None
        assert pod.basis.shape[0] == 60
        assert pod.basis.shape[1] <= 5
        assert pod.n_snapshots == 30

    def test_batch_update(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        X = self._make_snapshots()
        pod = IncrementalPOD(n_modes=5)
        pod.batch_update(X)  # (n_space, n_snap) auto-detected
        assert pod.n_snapshots == 30

    def test_batch_update_row_major(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 60))  # (n_snap, n_space)
        pod = IncrementalPOD(n_modes=5)
        pod.batch_update(X)
        assert pod.n_snapshots == 20

    def test_project(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        X = self._make_snapshots()
        pod = IncrementalPOD(n_modes=5)
        pod.batch_update(X)
        coeffs = pod.project(X[:, 0])
        assert coeffs.shape == (pod.basis.shape[1],)

    def test_energy_fraction_sums_to_one(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        X = self._make_snapshots()
        pod = IncrementalPOD(n_modes=5)
        pod.batch_update(X)
        ef = pod.energy_fraction()
        assert abs(ef.sum() - 1.0) < 1e-10

    def test_invalid_n_modes_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        with pytest.raises(ValueError, match="n_modes"):
            IncrementalPOD(n_modes=0)

    def test_invalid_forget_factor_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        with pytest.raises(ValueError, match="forget_factor"):
            IncrementalPOD(forget_factor=0.0)

    def test_update_non_1d_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        pod = IncrementalPOD(n_modes=3)
        with pytest.raises(ValueError, match="1D"):
            pod.update(np.zeros((5, 3)))

    def test_update_shape_mismatch_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        pod = IncrementalPOD(n_modes=3)
        pod.update(np.zeros(50))
        with pytest.raises(ValueError, match="shape"):
            pod.update(np.zeros(60))

    def test_batch_update_non_2d_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        pod = IncrementalPOD(n_modes=3)
        with pytest.raises(ValueError, match="2D"):
            pod.batch_update(np.zeros(50))

    def test_project_before_update_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        pod = IncrementalPOD(n_modes=3)
        with pytest.raises(RuntimeError, match="update"):
            pod.project(np.zeros(50))

    def test_energy_fraction_before_update_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        pod = IncrementalPOD(n_modes=3)
        with pytest.raises(RuntimeError, match="update"):
            pod.energy_fraction()

    def test_zero_snapshot_basis(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        pod = IncrementalPOD(n_modes=3)
        pod.update(np.zeros(50))  # zero snapshot
        assert pod.is_fitted
        assert pod.n_snapshots == 1

    def test_forget_factor_effect(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD

        X = self._make_snapshots()
        pod = IncrementalPOD(n_modes=5, forget_factor=0.9)
        pod.batch_update(X)
        assert pod.is_fitted

    def test_public_import_paths(self) -> None:
        from naviertwin.core.dimensionality_reduction import MRPOD as TopMRPOD
        from naviertwin.core.dimensionality_reduction import IncrementalPOD as TopIncrementalPOD
        from naviertwin.core.dimensionality_reduction.linear import MRPOD as LinearMRPOD
        from naviertwin.core.dimensionality_reduction.linear import (
            IncrementalPOD as LinearIncrementalPOD,
        )

        assert TopIncrementalPOD is LinearIncrementalPOD
        assert TopMRPOD is LinearMRPOD
