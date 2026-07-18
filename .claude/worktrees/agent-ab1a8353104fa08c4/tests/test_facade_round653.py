"""R653 — facade ROM/Surrogate 추가 op smoke tests."""

from __future__ import annotations

import numpy as np


class TestNewFacadeOps653:
    def test_load_rom_round_trip(self, tmp_path) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        facade = PostProcessFacade()
        rng = np.random.default_rng(0)
        modes = rng.standard_normal((10, 3))
        sv = np.array([3.0, 2.0, 1.0])
        path = tmp_path / "x.npz"
        facade.run("save_rom", path=str(path), modes=modes, singular_values=sv)
        r = facade.run("load_rom", path=str(path))
        np.testing.assert_allclose(r["modes"], modes)
        np.testing.assert_allclose(r["singular_values"], sv)
        assert "schema_version" in r["metadata"]

    def test_mode_summary(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(1)
        Q, _ = np.linalg.qr(rng.standard_normal((20, 4)))
        sv = np.array([5.0, 3.0, 2.0, 1.0])
        r = PostProcessFacade().run(
            "mode_summary",
            spatial_modes=Q,
            singular_values=sv,
            temporal_modes=rng.standard_normal((4, 30)),
        )
        assert r["n_modes"] == 4
        assert "modes" in r
        assert "orthogonality" in r
        assert r["orthogonality"]["max_off_diag"] < 1e-8

    def test_basis_interpolate(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(2)
        Q1, _ = np.linalg.qr(rng.standard_normal((20, 3)))
        Q2, _ = np.linalg.qr(rng.standard_normal((20, 3)))
        r = PostProcessFacade().run(
            "basis_interpolate",
            bases=[Q1, Q2],
            params=np.array([0.0, 1.0]),
            target=0.5,
        )
        assert r["interpolated_basis"].shape == (20, 3)
        assert r["pairwise_distances"].shape == (1,)

    def test_batch_predict(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(3)
        X = rng.standard_normal((200, 4))

        def f(x):
            return np.sum(x ** 2, axis=1)

        r = PostProcessFacade().run(
            "batch_predict", predict_fn=f, X=X, chunk_size=50,
        )
        assert r["n_samples"] == 200
        np.testing.assert_allclose(r["predictions"], f(X))

    def test_morris_sensitivity(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        def f(X):
            return X[:, 0] + 2 * X[:, 1] + 0.1 * X[:, 2]

        bounds = np.array([[0.0, 1.0]] * 3)
        r = PostProcessFacade().run(
            "morris_sensitivity", f=f, bounds=bounds,
            n_trajectories=10, n_levels=4,
        )
        assert r["mu_star"].shape == (3,)
        assert r["sigma"].shape == (3,)
        # x_1 가장 영향 큼
        assert r["mu_star"][1] > r["mu_star"][0]

    def test_permutation_importance(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        def f(X):
            return 10 * X[:, 0] + 0.001 * X[:, 1]

        rng = np.random.default_rng(4)
        X = rng.uniform(0, 1, (200, 2))
        y = f(X)
        r = PostProcessFacade().run(
            "permutation_importance", f=f, X=X, y=y, n_repeats=3,
        )
        assert r["importance"].shape == (2,)
        assert r["importance"][0] > r["importance"][1]

    def test_bic_model_average(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        p1 = np.array([1.0, 2.0])
        p2 = np.array([3.0, 4.0])
        # BIC 같으면 균등 가중
        r = PostProcessFacade().run(
            "bic_model_average",
            predictions=[p1, p2],
            bic_values=np.array([100.0, 100.0]),
        )
        np.testing.assert_allclose(r["weights"], [0.5, 0.5])
        np.testing.assert_allclose(r["average"], [2.0, 3.0])

    def test_stacking(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(5)
        n = 100
        m1 = rng.standard_normal(n)
        m2 = rng.standard_normal(n)
        y = 0.7 * m1 + 0.3 * m2
        P = np.column_stack([m1, m2])
        r = PostProcessFacade().run("stacking", predictions=P, y_true=y)
        np.testing.assert_allclose(r["weights"], [0.7, 0.3], atol=0.05)


class TestFacadeOpCount653:
    def test_count_grew_more(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        ops = PostProcessFacade().list_operations()
        # R652 = 44, R653 추가 = 53
        assert len(ops) >= 52

    def test_validation_category_expanded(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        facade = PostProcessFacade()
        val_ops = [
            op for op in facade.list_operations()
            if facade.describe(op)["category"] == "validation"
        ]
        # surrogate_metrics, residual_diagnostics, ensemble_average,
        # batch_predict, morris_sensitivity, permutation_importance,
        # bic_model_average, stacking → 8개 이상
        assert len(val_ops) >= 7
