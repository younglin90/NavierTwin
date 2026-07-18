"""Round 645 — POD mode inspection / packaging utilities."""

from __future__ import annotations

import numpy as np
import pytest


class TestModeSummary:
    def test_basic_keys(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            mode_summary,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 30))
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        info = mode_summary(U[:, :5], s[:5], Vt[:5, :])
        assert len(info) == 5
        for k in ("index", "spatial", "singular_value", "energy_fraction"):
            assert k in info[0]

    def test_temporal_optional(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            mode_summary,
        )

        rng = np.random.default_rng(1)
        U, _ = np.linalg.qr(rng.standard_normal((20, 3)))
        s = np.array([5.0, 3.0, 1.0])
        info = mode_summary(U, s)
        assert "temporal" not in info[0]

    def test_energy_fraction_sum_one(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            mode_summary,
        )

        s = np.array([5.0, 3.0, 1.0])
        U, _ = np.linalg.qr(np.random.default_rng(2).standard_normal((10, 3)))
        info = mode_summary(U, s)
        total = sum(i["energy_fraction"] for i in info)
        np.testing.assert_allclose(total, 1.0, atol=1e-10)

    def test_invalid_shape_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            mode_summary,
        )

        with pytest.raises(ValueError, match="spatial_modes"):
            mode_summary(np.zeros((10, 3)), np.zeros(5))

    def test_invalid_temporal_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            mode_summary,
        )

        U = np.random.default_rng(3).standard_normal((10, 3))
        with pytest.raises(ValueError, match="temporal_modes"):
            mode_summary(U, np.array([1.0, 1.0, 1.0]),
                          temporal_modes=np.zeros((4, 5)))


class TestModeOrthogonality:
    def test_orthogonal_zero_offdiag(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            mode_orthogonality,
        )

        rng = np.random.default_rng(4)
        U, _ = np.linalg.qr(rng.standard_normal((20, 4)))
        diag = mode_orthogonality(U)
        assert diag["max_off_diag"] < 1e-10
        assert diag["max_diag_dev"] < 1e-10

    def test_non_orthogonal_high(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            mode_orthogonality,
        )

        # 두 모드가 동일 → off-diag = 1
        U = np.tile(np.eye(5)[:, [0]], (1, 2))
        diag = mode_orthogonality(U)
        assert diag["max_off_diag"] > 0.5

    def test_invalid_shape(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            mode_orthogonality,
        )

        with pytest.raises(ValueError, match="2D"):
            mode_orthogonality(np.zeros(10))


class TestProjectSnapshot:
    def test_1d_snapshot(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            project_snapshot,
        )

        rng = np.random.default_rng(5)
        Q, _ = np.linalg.qr(rng.standard_normal((20, 4)))
        coeffs = np.array([1.0, 2.0, 3.0, 4.0])
        snap = Q @ coeffs
        result = project_snapshot(snap, Q)
        np.testing.assert_allclose(result, coeffs, atol=1e-10)

    def test_2d_snapshot(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            project_snapshot,
        )

        rng = np.random.default_rng(6)
        Q, _ = np.linalg.qr(rng.standard_normal((30, 3)))
        snaps = rng.standard_normal((30, 5))
        result = project_snapshot(snaps, Q)
        assert result.shape == (3, 5)

    def test_with_mean(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            project_snapshot,
        )

        Q, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((20, 3)))
        mean = np.full(20, 5.0)
        snap = Q @ np.array([1.0, 1.0, 1.0]) + mean
        result = project_snapshot(snap, Q, mean=mean)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0], atol=1e-10)

    def test_invalid_basis_ndim(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            project_snapshot,
        )

        with pytest.raises(ValueError, match="2D"):
            project_snapshot(np.zeros(10), np.zeros(10))

    def test_length_mismatch_1d(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            project_snapshot,
        )

        with pytest.raises(ValueError, match="length"):
            project_snapshot(np.zeros(8), np.zeros((10, 3)))

    def test_invalid_snap_ndim(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            project_snapshot,
        )

        with pytest.raises(ValueError, match="1D or 2D"):
            project_snapshot(np.zeros((10, 5, 3)), np.zeros((10, 3)))

    def test_2d_row_mismatch(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            project_snapshot,
        )

        with pytest.raises(ValueError, match="rows"):
            project_snapshot(np.zeros((8, 5)), np.zeros((10, 3)))


class TestTimeCoeffStats:
    def test_basic(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            time_coefficient_statistics,
        )

        rng = np.random.default_rng(8)
        coeffs = rng.standard_normal((100, 4))
        stats = time_coefficient_statistics(coeffs)
        for k in ("mean", "std", "range", "peak_freq"):
            assert k in stats
            assert stats[k].shape == (4,)

    def test_auto_transpose(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            time_coefficient_statistics,
        )

        rng = np.random.default_rng(9)
        # (r, n_t)로 입력
        coeffs = rng.standard_normal((4, 100))
        stats = time_coefficient_statistics(coeffs)
        # auto-transpose → (n_t, r) → r=4
        assert stats["mean"].shape == (4,)

    def test_invalid_ndim(self) -> None:
        from naviertwin.core.dimensionality_reduction.mode_inspection import (
            time_coefficient_statistics,
        )

        with pytest.raises(ValueError, match="2D"):
            time_coefficient_statistics(np.zeros(50))
