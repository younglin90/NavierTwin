"""Round 74 — Pipeline 자동 튜닝 (hyperopt)."""

from __future__ import annotations

import numpy as np
import pytest


class TestAutoTunePipeline:
    def test_random_backend(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.pipeline_hyperopt import auto_tune_pipeline

        rng = np.random.default_rng(0)
        X = rng.standard_normal((25, 16))
        params = np.linspace(0, 1, 16).reshape(-1, 1)

        result = auto_tune_pipeline(
            X, params,
            n_modes_range=(2, 5),
            surrogate_kinds=("kriging", "rbf"),
            n_trials=5,
            val_ratio=0.25,
            backend="random",
            seed=0,
        )
        assert "best_params" in result
        assert result["best_params"]["n_modes"] >= 2
        assert result["best_params"]["n_modes"] <= 5
        assert result["best_params"]["surrogate_kind"] in ("kriging", "rbf")
        assert len(result["history"]) == 5
        assert np.isfinite(result["best_rmse"])

    def test_best_rmse_matches_history(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.pipeline_hyperopt import auto_tune_pipeline

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 12))
        params = np.linspace(0, 1, 12).reshape(-1, 1)

        result = auto_tune_pipeline(
            X, params,
            n_modes_range=(1, 4),
            n_trials=4,
            backend="random",
            seed=1,
        )
        values = [h["value"] for h in result["history"]]
        assert result["best_rmse"] == min(values)
