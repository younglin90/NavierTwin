"""Round 51 — BoTorch BO 테스트."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("botorch", reason="BoTorch 필요")


class TestBoTorchBO:
    def test_qei_converges(self) -> None:
        from naviertwin.core.optimization.bayesian_opt_botorch import (
            BoTorchBayesianOpt,
        )

        def obj(x: np.ndarray) -> float:
            return float((x[0] - 0.3) ** 2 + (x[1] + 0.2) ** 2)

        opt = BoTorchBayesianOpt(
            bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
            n_initial=5, max_iter=5, q=2, acquisition="qei", seed=0,
        )
        x_best, f_best = opt.minimize(obj)
        assert f_best < 0.3
        assert x_best.shape == (2,)

    def test_ucb(self) -> None:
        from naviertwin.core.optimization.bayesian_opt_botorch import (
            BoTorchBayesianOpt,
        )

        def obj(x: np.ndarray) -> float:
            return float(np.sum(x ** 2))

        opt = BoTorchBayesianOpt(
            bounds=np.array([[-1.0, 1.0]]),
            n_initial=4, max_iter=4, q=1, acquisition="ucb", seed=0,
        )
        _, f_best = opt.minimize(obj)
        assert f_best < 0.5

    def test_invalid_acquisition(self) -> None:
        from naviertwin.core.optimization.bayesian_opt_botorch import (
            BoTorchBayesianOpt,
        )

        def obj(x: np.ndarray) -> float:
            return float(x[0] ** 2)

        opt = BoTorchBayesianOpt(
            bounds=np.array([[-1.0, 1.0]]),
            n_initial=3, max_iter=1, acquisition="foo",
        )
        with pytest.raises(ValueError, match="acquisition"):
            opt.minimize(obj)
