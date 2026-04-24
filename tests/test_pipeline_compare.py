"""Round 75 — 다중 모델 비교 백엔드."""

from __future__ import annotations

import numpy as np
import pytest


class TestCompareModels:
    def test_basic_compare(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.pipeline_compare import (
            compare_models,
            rank_table,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((25, 16))
        P = np.linspace(0, 1, 16).reshape(-1, 1)

        configs = [
            ("pod", 3, "kriging"),
            ("pod", 3, "rbf"),
            ("pod", 5, "kriging"),
        ]
        rows = compare_models(X, P, configs, seed=0)
        assert len(rows) == 3
        # 정렬 검증
        rmses = [r["rmse"] for r in rows]
        assert rmses == sorted(rmses)
        # 필수 필드
        for r in rows:
            assert {"rmse", "r2", "train_time_s", "reducer_kind", "n_modes"} <= r.keys()
            assert r["train_time_s"] >= 0

        txt = rank_table(rows)
        assert "rank" in txt
        assert "pod" in txt

    def test_failed_config_captured(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.pipeline_compare import compare_models

        rng = np.random.default_rng(0)
        X = rng.standard_normal((15, 10))
        P = np.linspace(0, 1, 10).reshape(-1, 1)

        # 잘못된 surrogate_kind → 실패 → error 필드 기록
        configs = [
            ("pod", 3, "kriging"),
            ("pod", 3, "nonexistent_surrogate"),
        ]
        rows = compare_models(X, P, configs, seed=0)
        assert len(rows) == 2
        failed = [r for r in rows if r["surrogate_kind"] == "nonexistent_surrogate"]
        assert len(failed) == 1
        assert "error" in failed[0]
        assert failed[0]["rmse"] == float("inf")
