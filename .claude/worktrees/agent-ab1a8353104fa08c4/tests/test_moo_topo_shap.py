"""v5.0.0 NSGA-II + SIMP + Kernel SHAP 테스트."""

from __future__ import annotations

import numpy as np


class TestNSGA2:
    def test_pareto_front_nontrivial(self) -> None:
        from naviertwin.core.optimization.moo_optimizer import NSGA2

        def obj(x: np.ndarray) -> list[float]:
            f1 = float((x[0] - 0.3) ** 2 + (x[1] + 0.2) ** 2)
            f2 = float((x[0] + 0.5) ** 2 + (x[1] - 0.4) ** 2)
            return [f1, f2]

        nsga = NSGA2(
            bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
            n_obj=2, pop_size=20, n_gen=15, seed=0,
        )
        P, F = nsga.optimize(obj)
        assert P.shape[0] >= 2
        assert F.shape[1] == 2
        # 파레토 해 쌍 비지배 확인
        from naviertwin.core.optimization.moo_optimizer import _dominates

        for i in range(len(F)):
            for j in range(len(F)):
                if i == j:
                    continue
                assert not _dominates(F[i], F[j])


class TestSIMP:
    def test_density_range(self) -> None:
        from naviertwin.core.optimization.topology_opt import simp_2d

        rho = simp_2d(nx=8, ny=5, vol_frac=0.5, penal=3.0, n_iter=5)
        assert rho.shape == (5, 8)
        assert 0.0 < rho.min() <= rho.max() <= 1.0
        vol = float(rho.mean())
        assert abs(vol - 0.5) < 0.25


class TestKernelSHAP:
    def test_linear_model_contributions(self) -> None:
        from naviertwin.core.explainability.shap_explainer import KernelSHAP

        # f = 3 x0 + 2 x1 + 5 x2  → 기여도가 가중치에 비례해야
        def f(X: np.ndarray) -> np.ndarray:
            return 3.0 * X[:, 0] + 2.0 * X[:, 1] + 5.0 * X[:, 2]

        rng = np.random.default_rng(0)
        bg = rng.standard_normal((50, 3))
        x = np.array([[1.0, 1.0, 1.0]])
        expl = KernelSHAP(f, bg, n_samples=500, seed=0)
        phi = expl.explain(x)[0]

        # 기여도 합 ≈ f(x) - f(평균)
        fx = float(f(x)[0])
        fmean = float(f(bg).mean())
        assert abs(phi.sum() - (fx - fmean)) < 1.5
        # x2 (계수 5) 는 상위 2개 안에 들어야 함 — MC Shapley 분산 허용
        top_two = set(np.argsort(np.abs(phi))[-2:])
        assert 2 in top_two
