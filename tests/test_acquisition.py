"""Round 636 — active learning acquisition functions: EI, PoI, UCB, TS, batch."""

from __future__ import annotations

import numpy as np
import pytest


class TestEI:
    def test_better_mu_higher_ei(self) -> None:
        from naviertwin.core.active_learning.acquisition import expected_improvement

        mu = np.array([1.0, 0.5, 0.1])
        sigma = np.array([0.3, 0.3, 0.3])
        ei = expected_improvement(mu, sigma, y_best=0.5, minimize=True)
        # μ가 작을수록 (더 좋음) EI가 큼
        assert ei[2] > ei[0]

    def test_zero_sigma(self) -> None:
        from naviertwin.core.active_learning.acquisition import expected_improvement

        mu = np.array([0.3, 0.7])
        sigma = np.array([0.0, 0.0])
        ei = expected_improvement(mu, sigma, y_best=0.5, minimize=True)
        # σ=0: EI = max(y_best - μ, 0)
        np.testing.assert_allclose(ei, [0.2, 0.0])

    def test_maximize_mode(self) -> None:
        from naviertwin.core.active_learning.acquisition import expected_improvement

        mu = np.array([1.0, 5.0])
        sigma = np.array([0.5, 0.5])
        ei = expected_improvement(mu, sigma, y_best=2.0, minimize=False)
        # μ 큰 쪽이 EI 더 큼
        assert ei[1] > ei[0]

    def test_negative_sigma_raises(self) -> None:
        from naviertwin.core.active_learning.acquisition import expected_improvement

        with pytest.raises(ValueError, match="sigma"):
            expected_improvement(
                np.array([1.0]), np.array([-0.1]), y_best=0.0,
            )

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.active_learning.acquisition import expected_improvement

        with pytest.raises(ValueError, match="shape"):
            expected_improvement(
                np.array([1.0, 2.0]), np.array([0.1]), y_best=0.0,
            )


class TestPoI:
    def test_in_range(self) -> None:
        from naviertwin.core.active_learning.acquisition import (
            probability_of_improvement,
        )

        rng = np.random.default_rng(0)
        mu = rng.standard_normal(20)
        sigma = np.abs(rng.standard_normal(20)) + 0.1
        p = probability_of_improvement(mu, sigma, y_best=0.0)
        assert np.all((0 <= p) & (p <= 1))

    def test_negative_sigma_raises(self) -> None:
        from naviertwin.core.active_learning.acquisition import (
            probability_of_improvement,
        )

        with pytest.raises(ValueError, match="sigma"):
            probability_of_improvement(
                np.array([1.0]), np.array([-0.1]), y_best=0.0,
            )

    def test_maximize(self) -> None:
        from naviertwin.core.active_learning.acquisition import (
            probability_of_improvement,
        )

        mu = np.array([1.0, 5.0])
        sigma = np.array([0.5, 0.5])
        # 큰 μ가 더 좋음 (max)
        p = probability_of_improvement(mu, sigma, y_best=2.0, minimize=False)
        assert p[1] > p[0]


class TestUCB:
    def test_basic(self) -> None:
        from naviertwin.core.active_learning.acquisition import (
            upper_confidence_bound,
        )

        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5, 0.1])
        ucb = upper_confidence_bound(mu, sigma, kappa=2.0)
        # mu=1, sigma=0.5 → UCB=2.0; mu=2, sigma=0.1 → UCB=2.2
        np.testing.assert_allclose(ucb, [2.0, 2.2])

    def test_kappa_zero_returns_mu(self) -> None:
        from naviertwin.core.active_learning.acquisition import (
            upper_confidence_bound,
        )

        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5, 0.1])
        ucb = upper_confidence_bound(mu, sigma, kappa=0.0)
        np.testing.assert_array_equal(ucb, mu)

    def test_negative_kappa_raises(self) -> None:
        from naviertwin.core.active_learning.acquisition import (
            upper_confidence_bound,
        )

        with pytest.raises(ValueError, match="kappa"):
            upper_confidence_bound(np.array([1.0]), np.array([0.1]), kappa=-1.0)


class TestLCB:
    def test_basic(self) -> None:
        from naviertwin.core.active_learning.acquisition import (
            lower_confidence_bound,
        )

        mu = np.array([1.0])
        sigma = np.array([0.5])
        lcb = lower_confidence_bound(mu, sigma, kappa=2.0)
        np.testing.assert_allclose(lcb, [0.0])


class TestThompson:
    def test_returns_finite(self) -> None:
        from naviertwin.core.active_learning.acquisition import thompson_sample

        mu = np.array([0.0, 1.0, 2.0])
        sigma = np.array([0.1, 0.2, 0.3])
        s = thompson_sample(mu, sigma, seed=0)
        assert s.shape == mu.shape
        assert np.isfinite(s).all()

    def test_deterministic_with_seed(self) -> None:
        from naviertwin.core.active_learning.acquisition import thompson_sample

        mu = np.zeros(5)
        sigma = np.ones(5)
        s1 = thompson_sample(mu, sigma, seed=42)
        s2 = thompson_sample(mu, sigma, seed=42)
        np.testing.assert_array_equal(s1, s2)


class TestMaxVariance:
    def test_basic(self) -> None:
        from naviertwin.core.active_learning.acquisition import max_variance_query

        s = np.array([0.1, 0.5, 0.3, 0.8])
        idx = max_variance_query(s, n_select=2)
        # 분산 큰 순 [3, 1]
        np.testing.assert_array_equal(idx, [3, 1])

    def test_invalid_n_select(self) -> None:
        from naviertwin.core.active_learning.acquisition import max_variance_query

        with pytest.raises(ValueError, match="n_select"):
            max_variance_query(np.zeros(5), n_select=10)


class TestQueryByCommittee:
    def test_disagreement_picks_max_var(self) -> None:
        from naviertwin.core.active_learning.acquisition import query_by_committee

        # 후보 0, 1, 2: 위원들이 0번에서 일치, 2번에서 불일치
        P = np.array([
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 5.0],
            [1.0, 2.0, -3.0],
        ])
        idx = query_by_committee(P, n_select=1)
        assert idx[0] == 2

    def test_invalid_predictions_ndim(self) -> None:
        from naviertwin.core.active_learning.acquisition import query_by_committee

        with pytest.raises(ValueError, match="2D"):
            query_by_committee(np.zeros(10), n_select=1)


class TestGreedyBatch:
    def test_no_distance_constraint(self) -> None:
        from naviertwin.core.active_learning.acquisition import (
            greedy_batch_acquisition,
        )

        acq = np.array([0.1, 0.5, 0.3, 0.8])
        cand = np.random.default_rng(0).standard_normal((4, 2))
        idx = greedy_batch_acquisition(acq, cand, batch_size=2)
        # 가장 큰 두 acq → [3, 1]
        assert set(idx.tolist()) == {3, 1}

    def test_with_min_distance(self) -> None:
        from naviertwin.core.active_learning.acquisition import (
            greedy_batch_acquisition,
        )

        # 가까운 두 점은 모두 선택 안 됨
        acq = np.array([0.9, 0.85, 0.1])
        cand = np.array([[0.0, 0.0], [0.01, 0.01], [10.0, 10.0]])
        idx = greedy_batch_acquisition(acq, cand, batch_size=2, min_distance=1.0)
        # 0번과 2번 (멀리)
        assert set(idx.tolist()) == {0, 2}

    def test_invalid_batch_size(self) -> None:
        from naviertwin.core.active_learning.acquisition import (
            greedy_batch_acquisition,
        )

        with pytest.raises(ValueError, match="batch_size"):
            greedy_batch_acquisition(
                np.zeros(5), np.zeros((5, 2)), batch_size=0,
            )

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.active_learning.acquisition import (
            greedy_batch_acquisition,
        )

        with pytest.raises(ValueError, match="candidates"):
            greedy_batch_acquisition(
                np.zeros(5), np.zeros((4, 2)), batch_size=1,
            )
