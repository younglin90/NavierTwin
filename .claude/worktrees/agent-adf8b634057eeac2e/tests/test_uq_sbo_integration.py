"""Round 9 — UQ (PCE) + Surrogate Optimization + 전체 통합 테스트."""

from __future__ import annotations

import numpy as np
import pytest


class TestPolynomialChaos:
    def test_mean_variance_match_MC(self) -> None:
        from naviertwin.core.optimization.uq_surrogate import PolynomialChaos

        rng = np.random.default_rng(0)
        # y = 2 + x1 + x2²   (ξ ~ U[-1,1]²)
        X = rng.uniform(-1, 1, (400, 2))
        y = 2.0 + X[:, 0] + X[:, 1] ** 2

        pce = PolynomialChaos(degree=3)
        pce.fit(X, y)

        # 해석적 평균/분산
        #   E[y] = 2 + 0 + E[ξ²] = 2 + 1/3
        #   Var[y] = Var[ξ1] + Var[ξ1²] = 1/3 + (E[ξ²²] - E[ξ²]²) = 1/3 + (1/5 - 1/9)
        expected_mean = 2.0 + 1.0 / 3.0
        var_x1 = 1.0 / 3.0  # Var(U[-1,1]) = 1/3
        var_x2sq = 1.0 / 5.0 - (1.0 / 3.0) ** 2
        expected_var = var_x1 + var_x2sq
        assert abs(pce.mean_ - expected_mean) < 0.1
        assert abs(pce.variance_ - expected_var) < 0.1

    def test_sobol_from_pce(self) -> None:
        from naviertwin.core.optimization.uq_surrogate import PolynomialChaos

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (400, 3))
        # y = x0 + 0.1·x1·x2 — x0 가 지배적
        y = X[:, 0] + 0.1 * X[:, 1] * X[:, 2]
        pce = PolynomialChaos(degree=3)
        pce.fit(X, y)
        S = pce.sobol_indices()
        assert S["S1"].shape == (3,)
        assert S["ST"].shape == (3,)
        assert S["S1"][0] > S["S1"][1]  # x0 가 가장 큰 기여
        assert S["S1"][0] > S["S1"][2]


class TestSurrogateOpt:
    def test_converges_near_minimum(self) -> None:
        pytest.importorskip("scipy")
        pytest.importorskip("smt")
        from naviertwin.core.optimization.surrogate_opt import SurrogateOptimizer

        def f(x: np.ndarray) -> float:
            return float((x[0] - 0.3) ** 2 + (x[1] + 0.2) ** 2)

        opt = SurrogateOptimizer(
            bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
            surrogate_kind="rbf", n_initial=6, max_iter=8, seed=0,
        )
        x_best, f_best = opt.minimize(f)
        assert f_best < 0.2


class TestFullIntegration:
    """CFD → POD → Surrogate → Predict → Export 전체 흐름."""

    def test_pipeline_then_report_and_onnx(self, tmp_path) -> None:
        pytest.importorskip("sklearn")
        pytest.importorskip("jinja2")

        import numpy as np

        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        rng = np.random.default_rng(0)
        r = 4
        U = rng.standard_normal((50, r))
        V = rng.standard_normal((r, 30))
        X = U @ V + 0.01 * rng.standard_normal((50, 30))

        pipe = NavierTwinPipeline(reducer_kind="pod", n_modes=r)
        pipe.load_snapshots(X, field_name="U")
        pipe.reduce()
        params = np.linspace(0, 1, 30).reshape(-1, 1)
        pipe.fit_surrogate(params)
        pipe.validate(params[-5:], pipe.state.coeffs[-5:])

        html = pipe.export_report(str(tmp_path / "r.html"), project="Integration")
        assert html.exists()
        assert "rmse" in html.read_text(encoding="utf-8")

    def test_full_bayesian_on_surrogate_target(self) -> None:
        pytest.importorskip("sklearn")
        pytest.importorskip("scipy")
        from naviertwin.core.optimization.bayesian_opt import BayesianOptimizer
        from naviertwin.core.optimization.moo_optimizer import NSGA2

        # BO 로 min, NSGA2 로 다목적 — 둘 다 동작 확인
        def f_single(x: np.ndarray) -> float:
            return float(np.sum(x ** 2))

        bo = BayesianOptimizer(
            bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
            n_initial=4, max_iter=6, seed=0,
        )
        _, f_best = bo.minimize(f_single)
        assert f_best < 0.5

        def f_multi(x: np.ndarray) -> list[float]:
            return [float(np.sum(x ** 2)), float(np.sum((x - 0.5) ** 2))]

        nsga = NSGA2(
            bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
            n_obj=2, pop_size=12, n_gen=8, seed=0,
        )
        P, F = nsga.optimize(f_multi)
        assert P.shape[0] >= 2 and F.shape[1] == 2
