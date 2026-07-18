"""v3.0.0 데이터 동화 + UQ + 최적화 테스트."""

from __future__ import annotations

import numpy as np
import pytest


class TestEnKF:
    def test_identity_observation_reduces_spread(self) -> None:
        from naviertwin.core.data_assimilation.enkf import EnKF

        rng = np.random.default_rng(0)
        n, N = 3, 100
        true = np.array([1.0, -0.5, 0.2])
        ens = rng.standard_normal((N, n))  # 큰 분산 초기 앙상블
        H = np.eye(n)
        R = 0.01 * np.eye(n)
        kf = EnKF(H=H, R=R)
        updated = kf.analysis(ens, true, rng=rng)
        # 업데이트 후 평균이 관측값에 가까워야
        assert np.linalg.norm(updated.mean(axis=0) - true) < 0.5

    def test_dimension_mismatch_raises(self) -> None:
        from naviertwin.core.data_assimilation.enkf import EnKF

        kf = EnKF(H=np.eye(3)[:2], R=np.eye(2))
        with pytest.raises(ValueError, match="observation"):
            kf.analysis(np.zeros((10, 3)), np.zeros(5))


class TestParticleFilter:
    def test_posterior_tracks_truth(self) -> None:
        from naviertwin.core.data_assimilation.particle_filter import ParticleFilter

        rng = np.random.default_rng(0)
        pf = ParticleFilter(n_particles=500, state_dim=2)
        pf.initialize(rng.standard_normal((500, 2)) * 3.0)
        true = np.array([1.0, -2.0])
        H = np.eye(2)
        R = 0.05 * np.eye(2)
        for _ in range(5):
            pf.update(true + 0.05 * rng.standard_normal(2), H, R)
        est = pf.estimate()
        assert np.linalg.norm(est - true) < 0.5


class TestSobol:
    def test_saltelli_shape(self) -> None:
        from naviertwin.core.sensitivity.sobol_analysis import saltelli_sample

        bounds = np.array([[0, 1], [0, 1], [0, 1]], dtype=float)
        X = saltelli_sample(bounds, n_base=64, seed=0)
        assert X.shape == (64 * (2 * 3 + 2), 3)

    def test_sobol_indices_ishigami(self) -> None:
        from naviertwin.core.sensitivity.sobol_analysis import (
            saltelli_sample,
            sobol_indices,
        )

        bounds = np.array([[-np.pi, np.pi]] * 3, dtype=float)
        X = saltelli_sample(bounds, n_base=1024, seed=0)

        # Ishigami — 알려진 민감도: X1, X2 기여 큼, X3 는 interaction 위주
        def ishigami(v: np.ndarray) -> np.ndarray:
            a, b = 7.0, 0.1
            return np.sin(v[:, 0]) + a * np.sin(v[:, 1]) ** 2 + b * v[:, 2] ** 4 * np.sin(v[:, 0])

        Y = ishigami(X)
        S = sobol_indices(Y, n_params=3)
        assert S["S1"].shape == (3,)
        assert S["ST"].shape == (3,)
        # X3 단독 S1 은 작아야 (interaction 주도)
        assert S["S1"][2] < 0.1


class TestMCPropagation:
    def test_mean_std_percentiles(self) -> None:
        from naviertwin.core.optimization.mc_propagation import propagate_mc

        rng = np.random.default_rng(0)
        X = rng.standard_normal((2000, 1))

        def f(x: np.ndarray) -> np.ndarray:
            return x[:, 0] ** 2

        stats = propagate_mc(f, X, percentiles=(5.0, 50.0, 95.0))
        assert float(stats["mean"][0]) == pytest.approx(1.0, abs=0.1)  # χ²₁ mean = 1
        assert 5.0 in stats["percentiles"]
        assert 95.0 in stats["percentiles"]


class TestBayesianOpt:
    def test_bo_converges_near_minimum(self) -> None:
        sklearn = pytest.importorskip("sklearn", reason="scikit-learn 필요")
        del sklearn
        from naviertwin.core.optimization.bayesian_opt import BayesianOptimizer

        def obj(x: np.ndarray) -> float:
            return float((x[0] - 0.3) ** 2 + (x[1] + 0.5) ** 2)

        opt = BayesianOptimizer(
            bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
            n_initial=5, max_iter=10, seed=0,
        )
        x_best, f_best = opt.minimize(obj)
        # 진짜 최소 0, (0.3, -0.5) 근방
        assert f_best < 0.3
