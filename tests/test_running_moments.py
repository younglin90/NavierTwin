"""Round 618 — Welford running moments + covariance + parallel merge."""

from __future__ import annotations

import numpy as np
import pytest


class TestRunningMoments:
    def test_scalar_mean(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningMoments

        rm = RunningMoments(shape=())
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rm.update(v)
        assert abs(rm.mean - 3.0) < 1e-12

    def test_scalar_variance(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningMoments

        rng = np.random.default_rng(0)
        x = rng.standard_normal(5000)
        rm = RunningMoments(shape=())
        for v in x:
            rm.update(float(v))
        # numpy 비교
        np.testing.assert_allclose(rm.variance_unbiased, x.var(ddof=1), rtol=1e-10)
        np.testing.assert_allclose(rm.mean, x.mean(), rtol=1e-10)

    def test_array_shape(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningMoments

        rng = np.random.default_rng(1)
        rm = RunningMoments(shape=(3,))
        samples = rng.standard_normal((100, 3))
        for s in samples:
            rm.update(s)
        np.testing.assert_allclose(rm.mean, samples.mean(axis=0), rtol=1e-10)

    def test_2d_shape(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningMoments

        rng = np.random.default_rng(2)
        rm = RunningMoments(shape=(4, 5))
        samples = rng.standard_normal((50, 4, 5))
        for s in samples:
            rm.update(s)
        np.testing.assert_allclose(rm.mean, samples.mean(axis=0), rtol=1e-10)

    def test_std(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningMoments

        rng = np.random.default_rng(3)
        x = rng.standard_normal(2000)
        rm = RunningMoments()
        for v in x:
            rm.update(float(v))
        # std 속성 = √(M2/n)
        np.testing.assert_allclose(rm.std, np.std(x), rtol=1e-10)

    def test_reset(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningMoments

        rm = RunningMoments(shape=(3,))
        rm.update([1.0, 2.0, 3.0])
        rm.update([4.0, 5.0, 6.0])
        rm.reset()
        assert rm.n == 0
        np.testing.assert_array_equal(rm.mean, np.zeros(3))

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningMoments

        rm = RunningMoments(shape=(3,))
        with pytest.raises(ValueError, match="sample shape"):
            rm.update([1.0, 2.0])

    def test_zero_samples_returns_zero_var(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningMoments

        rm = RunningMoments(shape=(3,))
        np.testing.assert_array_equal(rm.variance, np.zeros(3))
        np.testing.assert_array_equal(rm.variance_unbiased, np.zeros(3))


class TestMerge:
    def test_merge_equivalent_to_full(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningMoments

        rng = np.random.default_rng(4)
        x = rng.standard_normal(1000)

        # 분할 합산
        rm_a = RunningMoments()
        rm_b = RunningMoments()
        for v in x[:600]:
            rm_a.update(float(v))
        for v in x[600:]:
            rm_b.update(float(v))
        rm_a.merge(rm_b)

        # 직접 합산
        rm_full = RunningMoments()
        for v in x:
            rm_full.update(float(v))

        np.testing.assert_allclose(rm_a.mean, rm_full.mean, rtol=1e-10)
        np.testing.assert_allclose(rm_a.M2, rm_full.M2, rtol=1e-9)

    def test_merge_with_empty(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningMoments

        rm_a = RunningMoments()
        rm_a.update(1.0)
        rm_a.update(2.0)
        rm_b = RunningMoments()  # 빈 통계
        rm_a.merge(rm_b)
        assert rm_a.n == 2

    def test_merge_into_empty(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningMoments

        rm_a = RunningMoments()
        rm_b = RunningMoments()
        rm_b.update(1.0)
        rm_b.update(2.0)
        rm_a.merge(rm_b)
        assert rm_a.n == 2
        assert abs(rm_a.mean - 1.5) < 1e-12

    def test_merge_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningMoments

        rm_a = RunningMoments(shape=(3,))
        rm_b = RunningMoments(shape=(4,))
        with pytest.raises(ValueError, match="shape mismatch"):
            rm_a.merge(rm_b)


class TestRunningCovariance:
    def test_covariance(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningCovariance

        rng = np.random.default_rng(5)
        n = 5000
        # y = 2x + noise → cov(x, y) ≈ 2 var(x) = 2
        x = rng.standard_normal(n)
        y = 2 * x + 0.1 * rng.standard_normal(n)
        rc = RunningCovariance()
        for xv, yv in zip(x, y):
            rc.update(float(xv), float(yv))
        np.testing.assert_allclose(rc.covariance, 2.0, rtol=0.05)

    def test_correlation_strong(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningCovariance

        rng = np.random.default_rng(6)
        x = rng.standard_normal(2000)
        y = 3 * x + 0.01 * rng.standard_normal(2000)
        rc = RunningCovariance()
        for xv, yv in zip(x, y):
            rc.update(float(xv), float(yv))
        # 강한 양의 상관 → ~1
        assert rc.correlation > 0.99

    def test_correlation_independent(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningCovariance

        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(8)
        rc = RunningCovariance()
        for _ in range(2000):
            rc.update(rng1.standard_normal(), rng2.standard_normal())
        assert abs(rc.correlation) < 0.1

    def test_array_shape(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningCovariance

        rng = np.random.default_rng(9)
        rc = RunningCovariance(shape=(3,))
        for _ in range(500):
            x = rng.standard_normal(3)
            rc.update(x, 2 * x + 0.1 * rng.standard_normal(3))
        # 모든 성분 ~ 2
        np.testing.assert_allclose(rc.covariance, 2.0, atol=0.5)

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningCovariance

        rc = RunningCovariance(shape=(3,))
        with pytest.raises(ValueError, match="shape mismatch"):
            rc.update([1.0, 2.0], [1.0, 2.0])

    def test_zero_samples(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import RunningCovariance

        rc = RunningCovariance()
        assert rc.covariance == 0.0
        assert rc.correlation == 0.0


class TestOnePass:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.running_moments import online_mean_var

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m, v = online_mean_var(x)
        assert abs(m - 3.0) < 1e-12
        np.testing.assert_allclose(v, x.var(ddof=1))
