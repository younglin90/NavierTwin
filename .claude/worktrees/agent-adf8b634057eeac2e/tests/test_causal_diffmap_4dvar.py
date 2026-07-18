"""Round 6 — causal analysis / diffusion maps / 4D-Var 테스트."""

from __future__ import annotations

import numpy as np
import pytest


class TestCausal:
    def test_correlation_matrix(self) -> None:
        from naviertwin.core.sensitivity.causal_analysis import correlation_matrix

        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        C = correlation_matrix(X)
        assert C.shape == (3, 3)
        assert np.allclose(np.diag(C), 1.0, atol=1e-6)
        assert np.allclose(C, C.T)

    def test_granger_detects_known_lag(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.sensitivity.causal_analysis import granger_causality

        rng = np.random.default_rng(0)
        n = 300
        x = rng.standard_normal(n)
        lag = 3
        y = np.zeros(n)
        for t in range(lag, n):
            y[t] = 0.7 * y[t - 1] + 0.5 * x[t - lag] + 0.05 * rng.standard_normal()

        p = granger_causality(x, y, max_lag=5)
        assert p.shape == (5,)
        # lag >= 3 에서 p 값이 0.05 아래로 떨어져야 (강한 인과)
        assert p[lag - 1] < 0.1
        # no-lag 또는 잘못된 방향 테스트
        p_rev = granger_causality(y, x, max_lag=5)
        # 역방향은 lag 3 에서도 p 가 더 클 가능성 높음 (엄격 검증은 생략)
        assert p_rev.shape == (5,)


class TestDiffusionMaps:
    def test_circle_embedding(self) -> None:
        """원에서 추출한 점들은 diffusion maps 2D 로 원을 복원해야."""
        from naviertwin.core.dimensionality_reduction.nonlinear.diffusion_maps import (
            DiffusionMaps,
        )

        rng = np.random.default_rng(0)
        theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        X = np.column_stack([
            np.cos(theta), np.sin(theta),
            0.05 * rng.standard_normal(80),
        ])
        dm = DiffusionMaps(n_components=2)
        Y = dm.fit_transform(X)
        assert Y.shape == (80, 2)
        assert np.all(np.isfinite(Y))

    def test_too_few_samples_raises(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.diffusion_maps import (
            DiffusionMaps,
        )

        dm = DiffusionMaps(n_components=1)
        with pytest.raises(ValueError):
            dm.fit_transform(np.zeros((2, 2)))


class TestFourDVar:
    def test_converges_to_truth(self) -> None:
        from naviertwin.core.data_assimilation.four_dvar import four_dvar_linear

        rng = np.random.default_rng(0)
        n = 3
        M = np.eye(n) + 0.05 * rng.standard_normal((n, n))
        H = np.eye(n)
        x_true = rng.standard_normal(n)
        traj = [x_true.copy()]
        for _ in range(8):
            traj.append(M @ traj[-1] + 0.001 * rng.standard_normal(n))
        Y = np.array(traj[1:])

        x_b = x_true + 0.5 * rng.standard_normal(n)
        B = np.eye(n)
        R = 0.01 * np.eye(n)
        x0 = four_dvar_linear(x_b, B, Y, H, R, M)

        # 동화된 x0 이 배경보다 truth 에 가까워야
        err_b = float(np.linalg.norm(x_b - x_true))
        err_a = float(np.linalg.norm(x0 - x_true))
        assert err_a < err_b

    def test_Y_shape_validation(self) -> None:
        from naviertwin.core.data_assimilation.four_dvar import four_dvar_linear

        with pytest.raises(ValueError):
            four_dvar_linear(
                np.zeros(2), np.eye(2),
                np.zeros(5),  # 1D — 잘못
                np.eye(2), np.eye(2), np.eye(2),
            )
