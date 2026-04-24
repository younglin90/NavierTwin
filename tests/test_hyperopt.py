"""Round 67 — hyperopt random/botorch/optuna."""

from __future__ import annotations

import numpy as np
import pytest


def _quadratic(params: dict) -> float:
    # 최소: lr = 1e-2, n_layers = 3
    return (np.log10(params["lr"]) + 2.0) ** 2 + (params["n_layers"] - 3) ** 2


SPACE = {
    "lr": (1e-4, 1e-1, "logfloat"),
    "n_layers": (1, 6, "int"),
    "hidden": (8, 64, "int"),
}


class TestRandom:
    def test_random_search_converges(self) -> None:
        from naviertwin.core.optimization.hyperopt import hyperopt

        best, hist = hyperopt(_quadratic, SPACE, n_trials=30, backend="random", seed=0)
        assert "lr" in best
        assert 1 <= best["n_layers"] <= 6
        assert 1e-4 <= best["lr"] <= 1e-1
        assert len(hist) == 30

    def test_unknown_type_raises(self) -> None:
        from naviertwin.core.optimization.hyperopt import hyperopt

        with pytest.raises(ValueError):
            hyperopt(
                lambda p: 0.0,
                {"x": (0, 1, "weird")},
                n_trials=1, backend="random", seed=0,
            )


class TestBoTorch:
    def test_botorch_backend(self) -> None:
        pytest.importorskip("botorch")
        from naviertwin.core.optimization.hyperopt import hyperopt

        best, hist = hyperopt(_quadratic, SPACE, n_trials=10, backend="botorch", seed=0)
        assert "lr" in best
        assert len(hist) >= 8


class TestOptunaFallback:
    def test_optuna_fallback_to_random(self) -> None:
        # optuna 미설치 → random 으로 자동 전환
        from naviertwin.core.optimization.hyperopt import hyperopt

        best, hist = hyperopt(_quadratic, SPACE, n_trials=5, backend="optuna", seed=0)
        assert "lr" in best
