"""Round 160 — 마일스톤: R148-R159 import + 종합 e2e."""

from __future__ import annotations

import numpy as np
import pytest

R148_159 = [
    "naviertwin.core.export.torchscript_verify",
    "naviertwin.utils.device",
    "naviertwin.gui.signal_bridge",
    "naviertwin.gui.colormaps",
    "naviertwin.gui.plotting",
    "naviertwin.core.analysis.spatial_index",
    "naviertwin.core.data_assimilation.var4d_cost",
    "naviertwin.core.data_assimilation.enkf_simple",
    "naviertwin.core.sampling.monte_carlo",
    "naviertwin.core.uncertainty.pce_simple",
    "naviertwin.core.analysis.kdtree_wrapper",
    "naviertwin.core.tools.delaunay_2d",
]


class TestRound160:
    @pytest.mark.parametrize("m", R148_159)
    def test_importable(self, m: str) -> None:
        import importlib
        importlib.import_module(m)

    def test_sampling_to_pce_to_mc(self) -> None:
        """PCE fit 후 MC 로 평균 검증, KDTree 로 nearest 검증."""
        pytest.importorskip("scipy")
        from naviertwin.core.analysis.kdtree_wrapper import KDTreeIndex
        from naviertwin.core.sampling.monte_carlo import mc_integral
        from naviertwin.core.uncertainty.pce_simple import PCESimple

        rng = np.random.default_rng(0)
        xi = rng.uniform(-1, 1, size=(300, 1))
        y = 2.0 + xi[:, 0] - 0.5 * xi[:, 0] ** 2
        pce = PCESimple(order=3, family="legendre").fit(xi, y)
        # PCE 예측이 MC 기대값과 근사
        est, _ = mc_integral(
            lambda x: 2.0 + x - 0.5 * x ** 2, -1.0, 1.0, n=50000, seed=0,
        )
        true_mean = est / 2.0  # uniform[-1,1] 기대값
        y_mc = pce.predict(np.linspace(-1, 1, 500).reshape(-1, 1)).mean()
        assert abs(y_mc - true_mean) < 0.05

        # KD-tree 근접검색
        pts = rng.standard_normal((200, 3))
        kd = KDTreeIndex(pts)
        d, idx = kd.knn(np.array([[0.0, 0.0, 0.0]]), k=3)
        assert idx.shape == (1, 3)
        assert np.all(d[0] >= 0)

    def test_enkf_update_decreases_spread(self) -> None:
        from naviertwin.core.data_assimilation.enkf_simple import EnKFSimple

        rng = np.random.default_rng(0)
        ens = rng.multivariate_normal([0, 0], np.eye(2) * 2.0, size=300)
        kf = EnKFSimple(H=np.eye(2), R=np.eye(2) * 0.05)
        ens_post = kf.update(ens, z=np.array([1.0, 1.0]), rng=rng)
        # 분산 감소
        assert ens_post.var(axis=0).sum() < ens.var(axis=0).sum()
