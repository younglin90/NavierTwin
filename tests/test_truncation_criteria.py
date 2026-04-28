"""Round 629 — POD truncation criteria (energy, Eckart-Young, scree, AIC/BIC)."""

from __future__ import annotations

import numpy as np
import pytest


class TestEnergy:
    def test_basic(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            truncate_by_energy,
        )

        # σ = [10, 5, 1, 0.1] → 에너지 = [100, 25, 1, 0.01], 합 = 126.01
        s = np.array([10.0, 5.0, 1.0, 0.1])
        # 99% 보존 → 처음 3개 누적 = 126/126.01 ≈ 99.99%
        r = truncate_by_energy(s, fraction=0.99)
        assert r == 2

    def test_fraction_one(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            truncate_by_energy,
        )

        s = np.array([3.0, 2.0, 1.0])
        r = truncate_by_energy(s, fraction=1.0)
        assert r == 3

    def test_zero_singular_values(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            truncate_by_energy,
        )

        r = truncate_by_energy(np.zeros(5), fraction=0.99)
        assert r == 1

    def test_invalid_fraction(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            truncate_by_energy,
        )

        with pytest.raises(ValueError, match="fraction"):
            truncate_by_energy(np.ones(5), fraction=0.0)

    def test_empty_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            truncate_by_energy,
        )

        with pytest.raises(ValueError, match="empty"):
            truncate_by_energy(np.array([]))


class TestEckartYoung:
    def test_basic(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            truncate_by_eckart_young,
        )

        s = np.array([10.0, 5.0, 1.0, 0.1])
        r = truncate_by_eckart_young(s, rel_error=0.05)
        assert r >= 1

    def test_invalid_rel_error(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            truncate_by_eckart_young,
        )

        with pytest.raises(ValueError, match="rel_error"):
            truncate_by_eckart_young(np.ones(5), rel_error=0.0)


class TestScreeElbow:
    def test_clear_elbow(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            scree_elbow,
        )

        # 명확한 엘보우: 큰 3개 + 작은 7개
        s = np.array([10.0, 9.0, 8.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        elbow = scree_elbow(s)
        assert 1 <= elbow <= 5

    def test_too_few_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            scree_elbow,
        )

        with pytest.raises(ValueError, match="3"):
            scree_elbow(np.array([1.0, 2.0]))


class TestAICBIC:
    def test_aic_returns_int(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            truncate_by_aic,
        )

        s = np.array([5.0, 3.0, 1.0, 0.5, 0.1])
        r = truncate_by_aic(s, n_samples=100)
        assert isinstance(r, int)
        assert 1 <= r <= 5

    def test_bic_returns_int(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            truncate_by_bic,
        )

        s = np.array([5.0, 3.0, 1.0, 0.5, 0.1])
        r = truncate_by_bic(s, n_samples=100)
        assert isinstance(r, int)
        assert 1 <= r <= 5

    def test_invalid_n_samples_aic(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            truncate_by_aic,
        )

        with pytest.raises(ValueError, match="n_samples"):
            truncate_by_aic(np.ones(5), n_samples=0)

    def test_invalid_n_samples_bic(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            truncate_by_bic,
        )

        with pytest.raises(ValueError, match="n_samples"):
            truncate_by_bic(np.ones(5), n_samples=-1)


class TestRelativeError:
    def test_full_rank_zero(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            relative_l2_error,
        )

        s = np.array([3.0, 2.0, 1.0])
        err = relative_l2_error(s, r=3)
        assert err == 0.0

    def test_zero_rank_unity(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            relative_l2_error,
        )

        s = np.array([3.0, 2.0, 1.0])
        err = relative_l2_error(s, r=0)
        assert abs(err - 1.0) < 1e-12

    def test_invalid_r(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            relative_l2_error,
        )

        with pytest.raises(ValueError, match="r must be"):
            relative_l2_error(np.ones(5), r=10)


class TestCumulativeEnergy:
    def test_monotonic(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            cumulative_energy_curve,
        )

        s = np.array([5.0, 3.0, 2.0, 1.0])
        curve = cumulative_energy_curve(s)
        assert curve.shape == (4,)
        assert np.all(np.diff(curve) >= 0)
        assert abs(curve[-1] - 1.0) < 1e-12

    def test_zero_input(self) -> None:
        from naviertwin.core.dimensionality_reduction.truncation_criteria import (
            cumulative_energy_curve,
        )

        curve = cumulative_energy_curve(np.zeros(5))
        np.testing.assert_array_equal(curve, np.zeros(5))
