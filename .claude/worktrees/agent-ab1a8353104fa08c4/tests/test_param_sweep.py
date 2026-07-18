"""Round 78 — 파라미터 스윕 (LHS/Sobol/Halton/grid/random)."""

from __future__ import annotations

import numpy as np
import pytest


class TestParamSweep:
    @pytest.mark.parametrize("kind", ["lhs", "sobol", "halton", "random"])
    def test_bounds_respected(self, kind: str) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.sampling.param_sweep import generate_sweep

        bounds = [(0.1, 1.0), (100.0, 500.0), (-1.0, 1.0)]
        pts = generate_sweep(bounds, n_points=16, kind=kind, seed=0)
        assert pts.shape == (16, 3)
        for i, (lo, hi) in enumerate(bounds):
            assert pts[:, i].min() >= lo - 1e-10
            assert pts[:, i].max() <= hi + 1e-10

    def test_grid(self) -> None:
        from naviertwin.core.sampling.param_sweep import generate_sweep

        pts = generate_sweep([(0.0, 1.0), (0.0, 2.0)], n_points=9, kind="grid")
        assert pts.shape[1] == 2
        assert pts.shape[0] <= 9

    def test_reproducibility(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.sampling.param_sweep import generate_sweep

        a = generate_sweep([(0, 1), (0, 1)], n_points=8, kind="lhs", seed=42)
        b = generate_sweep([(0, 1), (0, 1)], n_points=8, kind="lhs", seed=42)
        assert np.allclose(a, b)
