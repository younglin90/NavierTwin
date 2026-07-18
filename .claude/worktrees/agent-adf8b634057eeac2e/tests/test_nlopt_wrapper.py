"""Round 53 — NLopt 래퍼 테스트."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nlopt", reason="nlopt 필요")


class TestNLopt:
    def test_bobyqa_converges(self) -> None:
        from naviertwin.core.optimization.nlopt_wrapper import nlopt_minimize

        def obj(x: np.ndarray) -> float:
            return float((x[0] - 1.5) ** 2 + (x[1] + 2.0) ** 2)

        x_best, f_best = nlopt_minimize(
            obj, x0=np.array([0.0, 0.0]),
            bounds=np.array([[-5.0, 5.0], [-5.0, 5.0]]),
            algorithm="LN_BOBYQA", max_eval=300,
        )
        assert f_best < 0.01
        assert abs(x_best[0] - 1.5) < 0.05
        assert abs(x_best[1] + 2.0) < 0.05

    def test_direct_global(self) -> None:
        from naviertwin.core.optimization.nlopt_wrapper import nlopt_minimize

        # 다중 국소 최소 함수: sin(x) + 0.1 x² — 글로벌 근처에서 DIRECT 수렴
        def obj(x: np.ndarray) -> float:
            return float(np.sin(x[0]) + 0.1 * x[0] ** 2)

        x_best, f_best = nlopt_minimize(
            obj, x0=np.array([5.0]),
            bounds=np.array([[-10.0, 10.0]]),
            algorithm="GN_DIRECT", max_eval=300,
        )
        assert f_best < 0.0  # sin(x) 음수 구간 찾아야

    def test_gradient_lbfgs(self) -> None:
        from naviertwin.core.optimization.nlopt_wrapper import nlopt_minimize

        def obj(x: np.ndarray) -> float:
            return float(np.sum(x ** 2))

        def grad(x: np.ndarray) -> np.ndarray:
            return 2 * x

        x_best, f_best = nlopt_minimize(
            obj, x0=np.array([3.0, -2.0]),
            bounds=np.array([[-5.0, 5.0], [-5.0, 5.0]]),
            algorithm="LD_LBFGS", max_eval=200, grad_fn=grad,
        )
        assert f_best < 1e-6

    def test_unknown_algorithm(self) -> None:
        from naviertwin.core.optimization.nlopt_wrapper import nlopt_minimize

        with pytest.raises(ValueError, match="algorithm"):
            nlopt_minimize(
                lambda x: float(x[0] ** 2),
                x0=np.array([1.0]),
                bounds=np.array([[-1.0, 1.0]]),
                algorithm="FOO_BAR",
            )

    def test_list_algorithms(self) -> None:
        from naviertwin.core.optimization.nlopt_wrapper import list_algorithms

        algos = list_algorithms()
        assert len(algos) >= 10
        assert "LN_BOBYQA" in algos
        assert "GN_DIRECT" in algos
