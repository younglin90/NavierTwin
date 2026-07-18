"""Round 260 — 마일스톤 R251-R259."""

from __future__ import annotations

import numpy as np
import pytest

R251_259 = [
    "naviertwin.core.tools.quadtree_amr",
    "naviertwin.core.sampling.poisson_disk",
    "naviertwin.core.neural.loss_library",
    "naviertwin.core.neural.activations",
    "naviertwin.core.solvers.fv_1d",
    "naviertwin.core.optimization.tangent_linear",
    "naviertwin.core.analysis.ftle",
    "naviertwin.core.optimization.halving",
    "naviertwin.core.analysis.dtw",
]


class TestRound260:
    @pytest.mark.parametrize("m", R251_259)
    def test_importable(self, m: str) -> None:
        import importlib
        importlib.import_module(m)

    def test_poisson_to_ftle(self) -> None:
        """Poisson-disk seeds → FTLE 없어도 pipeline 호환성만 검증."""
        from naviertwin.core.sampling.poisson_disk import poisson_disk_2d

        pts = poisson_disk_2d(1.0, 1.0, r=0.12, seed=0)
        assert pts.shape[1] == 2
        # min pairwise >= r
        D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        assert D.min() >= 0.12 - 1e-9

    def test_loss_activation_roundtrip(self) -> None:
        from naviertwin.core.neural.activations import sigmoid
        from naviertwin.core.neural.loss_library import mse

        rng = np.random.default_rng(0)
        x = rng.standard_normal(20)
        y = sigmoid(x)
        # self-MSE = 0
        assert mse(y, y) == 0.0
