"""Round 210 — 마일스톤 R201-R209."""

from __future__ import annotations

import numpy as np
import pytest

R201_209 = [
    "naviertwin.core.neural.unet_2d",
    "naviertwin.core.neural.attention_pool",
    "naviertwin.core.validation.image_metrics",
    "naviertwin.core.analysis.velocity_gradient",
    "naviertwin.core.digital_twin.pipeline_builder",
    "naviertwin.core.linalg.pcg",
    "naviertwin.core.dimensionality_reduction.linear.snapshot_pod_scratch",
    "naviertwin.core.monitoring.dashboard",
    "naviertwin.core.analysis.gmm",
]


class TestRound210:
    @pytest.mark.parametrize("m", R201_209)
    def test_importable(self, m: str) -> None:
        import importlib

        importlib.import_module(m)

    def test_snap_pod_dashboard(self) -> None:
        """snapshot POD → 에너지 dashboard aggregate."""
        from naviertwin.core.dimensionality_reduction.linear.snapshot_pod_scratch import (
            snapshot_pod,
        )
        from naviertwin.core.monitoring.dashboard import DashboardAggregator

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 12))
        _, sv, _ = snapshot_pod(X, k=5)
        d = DashboardAggregator()
        for v in sv:
            d.push("sigma", float(v))
        s = d.summary("sigma")
        assert s["count"] == 5
        assert s["max"] >= s["min"]

    def test_image_metrics_and_pipeline_builder(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.pipeline_builder import build_pipeline
        from naviertwin.core.validation.image_metrics import psnr

        rng = np.random.default_rng(0)
        pipe = build_pipeline({"n_modes": 3, "surrogate_kind": "rbf"})
        X = rng.standard_normal((15, 10))
        pipe.load_snapshots(X, field_name="U")
        pipe.reduce()
        pipe.fit_surrogate(np.linspace(0, 1, 10).reshape(-1, 1))
        pred = pipe.predict_field(np.array([[0.5]]))
        p = psnr(pred[:, 0], X[:, 5], data_range=float(X.max() - X.min()))
        assert np.isfinite(p)
