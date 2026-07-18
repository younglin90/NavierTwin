"""Round 42 — Langevin sampler + Euler-Maruyama."""

from __future__ import annotations

import numpy as np


class TestLangevin:
    def test_gaussian_recovery(self) -> None:
        """score(x) = -x → p ∝ exp(-0.5 x²), 표본 평균 ≈ 0."""
        from naviertwin.core.generative.langevin import langevin_sample

        rng = np.random.default_rng(0)

        def score(x: np.ndarray) -> np.ndarray:
            return -x

        traj = langevin_sample(
            score, x0=np.array([3.0]), n_steps=5000, step_size=0.02, rng=rng,
        )
        # 트래젝토리가 유한하고 초기값 3 에서 0 근처로 이동
        assert np.all(np.isfinite(traj))
        # 전체 평균은 ±1 이내
        assert abs(float(traj[1000:].mean())) < 1.0
        # 초기와 비교해 std 가 합리적 (주로 1 근처)
        assert 0.3 < float(traj[1000:].std()) < 3.0


class TestEulerMaruyama:
    def test_geometric_brownian_positive(self) -> None:
        """dX = μX dt + σX dW → 양의 궤적."""
        from naviertwin.core.generative.langevin import euler_maruyama

        rng = np.random.default_rng(0)
        mu, sigma = 0.1, 0.2

        def drift(t: float, x: np.ndarray) -> np.ndarray:
            return mu * x

        def diff(t: float, x: np.ndarray) -> np.ndarray:
            return sigma * x

        times, traj = euler_maruyama(
            drift, diff, x0=np.array([1.0]), t_span=(0, 1), n_steps=200, rng=rng,
        )
        assert traj.shape == (201, 1)
        # 단기간이므로 양수 유지
        assert np.all(traj > 0)
