"""R652 — facade ROM/Surrogate 확장 op smoke tests."""

from __future__ import annotations

import numpy as np


class TestNewFacadeOps:
    def test_helmholtz_decomp(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(0)
        u = rng.standard_normal((16, 16))
        v = rng.standard_normal((16, 16))
        r = PostProcessFacade().run("helmholtz_decomp", u=u, v=v)
        for k in ("solenoidal_u", "solenoidal_v", "irrotational_u", "irrotational_v"):
            assert k in r and r[k].shape == u.shape

    def test_rom_residual(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 30))
        Q, _ = np.linalg.qr(rng.standard_normal((30, 5)))
        r = PostProcessFacade().run("rom_residual", X=X, basis=Q)
        assert "residual" in r and "relative_residual" in r

    def test_rom_envelope(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(2)
        coeffs = rng.standard_normal((100, 5))
        new = coeffs.mean(axis=0)
        r = PostProcessFacade().run("rom_envelope", coeffs_train=coeffs, new_coeff=new)
        assert "envelope" in r
        assert "max_z" in r["envelope"]

    def test_subspace_drift(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(3)
        Q1, _ = np.linalg.qr(rng.standard_normal((20, 4)))
        Q2, _ = np.linalg.qr(Q1 + 0.1 * rng.standard_normal((20, 4)))
        r = PostProcessFacade().run("subspace_drift", basis_old=Q1, basis_new=Q2)
        assert r["drift_score"] >= 0

    def test_gappy_reconstruct(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(4)
        Q, _ = np.linalg.qr(rng.standard_normal((30, 4)))
        partial = Q @ np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.ones(30, dtype=bool)
        mask[10:20] = False
        r = PostProcessFacade().run(
            "gappy_reconstruct", basis=Q, partial=partial, mask=mask,
        )
        assert r["coefficients"].shape == (4,)
        assert r["reconstructed"].shape == (30,)

    def test_surrogate_metrics(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(5)
        y = rng.standard_normal(100)
        y_pred = y + 0.1 * rng.standard_normal(100)
        r = PostProcessFacade().run("surrogate_metrics", y_true=y, y_pred=y_pred)
        for k in ("rmse", "nrmse_range", "cv_rmse", "r2"):
            assert k in r

    def test_residual_diagnostics(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(6)
        residuals = rng.standard_normal(500)
        r = PostProcessFacade().run("residual_diagnostics", residuals=residuals)
        assert "diagnostic" in r
        assert "mean" in r["diagnostic"]
        assert "dw" in r["diagnostic"]

    def test_ensemble_average_uniform(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        p1 = np.array([1.0, 2.0])
        p2 = np.array([3.0, 4.0])
        r = PostProcessFacade().run("ensemble_average", predictions=[p1, p2])
        np.testing.assert_allclose(r["average"], [2.0, 3.0])
        np.testing.assert_allclose(r["variance"], [1.0, 1.0])

    def test_ensemble_average_weighted(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        p1 = np.array([1.0])
        p2 = np.array([3.0])
        r = PostProcessFacade().run(
            "ensemble_average",
            predictions=[p1, p2],
            weights=np.array([0.25, 0.75]),
        )
        np.testing.assert_allclose(r["average"], [2.5])

    def test_trajectory_clustering(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(7)
        coeffs = rng.standard_normal((100, 5))
        r = PostProcessFacade().run(
            "trajectory_clustering", coeffs=coeffs, window=20, n_clusters=3,
        )
        assert r["labels"].shape == (81,)
        assert r["centers"].shape == (3, 5)
        assert "silhouette" in r

    def test_acoustic_strouhal(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        r = PostProcessFacade().run("acoustic_strouhal", f=100.0, L=0.1, U=10.0)
        # St = 100 * 0.1 / 10 = 1
        assert abs(r["strouhal_number"] - 1.0) < 1e-10

    def test_save_rom(self, tmp_path) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(8)
        modes = rng.standard_normal((20, 5))
        sv = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
        path = tmp_path / "rom.npz"
        r = PostProcessFacade().run(
            "save_rom",
            path=str(path),
            modes=modes,
            singular_values=sv,
        )
        assert "saved_path" in r
        assert path.exists()


class TestFacadeOpCount:
    def test_count_grew(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        ops = PostProcessFacade().list_operations()
        # R651 = 33, R652 추가 = 44+ (acoustic 포함)
        assert len(ops) >= 43

    def test_new_categories(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        facade = PostProcessFacade()
        cats = {facade.describe(op)["category"] for op in facade.list_operations()}
        # R652 신규 카테고리
        assert "rom" in cats
        assert "validation" in cats

    def test_all_describable(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        facade = PostProcessFacade()
        for op in facade.list_operations():
            info = facade.describe(op)
            assert info["name"] == op
            assert info["category"]
            assert info["description"]
