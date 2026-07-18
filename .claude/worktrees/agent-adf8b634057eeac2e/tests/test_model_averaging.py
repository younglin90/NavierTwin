"""Round 635 — model averaging: equal/weighted/CV-based/BIC/stacking + ensemble variance."""

from __future__ import annotations

import numpy as np
import pytest


class TestEqualWeight:
    def test_basic(self) -> None:
        from naviertwin.core.surrogate.model_averaging import equal_weight_average

        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([3.0, 2.0, 1.0])
        avg = equal_weight_average([p1, p2])
        np.testing.assert_allclose(avg, [2.0, 2.0, 2.0])

    def test_single_model(self) -> None:
        from naviertwin.core.surrogate.model_averaging import equal_weight_average

        p = np.array([1.0, 2.0])
        np.testing.assert_array_equal(equal_weight_average([p]), p)

    def test_empty_raises(self) -> None:
        from naviertwin.core.surrogate.model_averaging import equal_weight_average

        with pytest.raises(ValueError, match="empty"):
            equal_weight_average([])


class TestWeightedAverage:
    def test_basic(self) -> None:
        from naviertwin.core.surrogate.model_averaging import weighted_average

        p1 = np.array([1.0, 1.0])
        p2 = np.array([3.0, 3.0])
        avg = weighted_average([p1, p2], weights=np.array([0.25, 0.75]))
        # 1*0.25 + 3*0.75 = 2.5
        np.testing.assert_allclose(avg, [2.5, 2.5])

    def test_auto_normalize(self) -> None:
        from naviertwin.core.surrogate.model_averaging import weighted_average

        p1 = np.array([1.0])
        p2 = np.array([3.0])
        # 가중치 합 ≠ 1이라도 정규화됨
        avg = weighted_average([p1, p2], weights=np.array([1.0, 3.0]))
        np.testing.assert_allclose(avg, [2.5])

    def test_weight_length_mismatch(self) -> None:
        from naviertwin.core.surrogate.model_averaging import weighted_average

        with pytest.raises(ValueError, match="weights"):
            weighted_average(
                [np.zeros(3), np.zeros(3)], weights=np.array([0.5, 0.3, 0.2]),
            )

    def test_zero_weights_raises(self) -> None:
        from naviertwin.core.surrogate.model_averaging import weighted_average

        with pytest.raises(ValueError, match="zero"):
            weighted_average([np.zeros(3)], weights=np.array([0.0]))

    def test_empty_raises(self) -> None:
        from naviertwin.core.surrogate.model_averaging import weighted_average

        with pytest.raises(ValueError, match="empty"):
            weighted_average([], weights=np.array([]))


class TestCVWeights:
    def test_smaller_error_higher_weight(self) -> None:
        from naviertwin.core.surrogate.model_averaging import cv_error_weights

        e = np.array([1.0, 2.0, 4.0])
        w = cv_error_weights(e)
        assert w[0] > w[1] > w[2]
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_negative_error_raises(self) -> None:
        from naviertwin.core.surrogate.model_averaging import cv_error_weights

        with pytest.raises(ValueError, match="non-negative"):
            cv_error_weights(np.array([1.0, -1.0]))

    def test_zero_errors(self) -> None:
        from naviertwin.core.surrogate.model_averaging import cv_error_weights

        # 모두 0 → 균등 weights
        w = cv_error_weights(np.zeros(3))
        np.testing.assert_allclose(w, 1.0 / 3.0)


class TestBICWeights:
    def test_lower_bic_higher_prob(self) -> None:
        from naviertwin.core.surrogate.model_averaging import bic_weights

        bic = np.array([100.0, 105.0, 200.0])
        w = bic_weights(bic)
        assert w[0] > w[1] > w[2]
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_equal_bics_uniform(self) -> None:
        from naviertwin.core.surrogate.model_averaging import bic_weights

        w = bic_weights(np.array([10.0, 10.0, 10.0]))
        np.testing.assert_allclose(w, 1.0 / 3.0)


class TestStackingLeastSquares:
    def test_recovers_true_weights(self) -> None:
        from naviertwin.core.surrogate.model_averaging import (
            stacking_least_squares,
        )

        rng = np.random.default_rng(0)
        n = 100
        # 두 모델: y = 0.7 m1 + 0.3 m2
        m1 = rng.standard_normal(n)
        m2 = rng.standard_normal(n)
        y = 0.7 * m1 + 0.3 * m2
        P = np.column_stack([m1, m2])
        w = stacking_least_squares(P, y)
        np.testing.assert_allclose(w, [0.7, 0.3], atol=1e-2)

    def test_nonnegative_clip(self) -> None:
        from naviertwin.core.surrogate.model_averaging import (
            stacking_least_squares,
        )

        rng = np.random.default_rng(1)
        n = 50
        m1 = rng.standard_normal(n)
        m2 = -m1 + 0.1 * rng.standard_normal(n)
        y = m1
        P = np.column_stack([m1, m2])
        w = stacking_least_squares(P, y, nonnegative=True)
        assert np.all(w >= 0)

    def test_predictions_2d_required(self) -> None:
        from naviertwin.core.surrogate.model_averaging import (
            stacking_least_squares,
        )

        with pytest.raises(ValueError, match="2D"):
            stacking_least_squares(np.zeros(10), np.zeros(10))

    def test_row_mismatch_raises(self) -> None:
        from naviertwin.core.surrogate.model_averaging import (
            stacking_least_squares,
        )

        with pytest.raises(ValueError, match="rows"):
            stacking_least_squares(np.zeros((10, 3)), np.zeros(8))

    def test_zero_weights_fallback_uniform(self) -> None:
        from naviertwin.core.surrogate.model_averaging import (
            stacking_least_squares,
        )

        # 강제로 음의 가중치만 → nonnegative 후 0 → 균등
        P = np.array([[-1.0, -2.0]] * 10)
        y = np.zeros(10)
        w = stacking_least_squares(P, y, nonnegative=True)
        # 모두 0 후 fallback → 균등
        np.testing.assert_allclose(w, [0.5, 0.5])


class TestEnsembleVariance:
    def test_zero_for_identical(self) -> None:
        from naviertwin.core.surrogate.model_averaging import ensemble_variance

        p = np.array([1.0, 2.0, 3.0])
        var = ensemble_variance([p, p, p])
        np.testing.assert_allclose(var, 0.0)

    def test_disagreement_positive(self) -> None:
        from naviertwin.core.surrogate.model_averaging import ensemble_variance

        p1 = np.array([1.0, 2.0])
        p2 = np.array([3.0, 0.0])
        var = ensemble_variance([p1, p2])
        # avg = [2, 1]; var = [(1-2)²+(3-2)²]/2 = 1
        np.testing.assert_allclose(var, [1.0, 1.0])

    def test_with_weights(self) -> None:
        from naviertwin.core.surrogate.model_averaging import ensemble_variance

        p1 = np.array([1.0])
        p2 = np.array([5.0])
        var = ensemble_variance([p1, p2], weights=np.array([0.9, 0.1]))
        # 가중치가 한 쪽 모델로 치우침 → 분산 작아짐
        var_eq = ensemble_variance([p1, p2])
        assert var[0] < var_eq[0]

    def test_empty_raises(self) -> None:
        from naviertwin.core.surrogate.model_averaging import ensemble_variance

        with pytest.raises(ValueError, match="empty"):
            ensemble_variance([])

    def test_zero_weights_raises(self) -> None:
        from naviertwin.core.surrogate.model_averaging import ensemble_variance

        with pytest.raises(ValueError, match="zero"):
            ensemble_variance([np.zeros(3)], weights=np.array([0.0]))
