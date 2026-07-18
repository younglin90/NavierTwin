"""R648 — PostProcessFacade smoke tests + GUI integration verification."""

from __future__ import annotations

import numpy as np
import pytest


class TestFacadeBasics:
    def test_list_operations_nonempty(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        facade = PostProcessFacade()
        ops = facade.list_operations()
        assert len(ops) >= 15
        assert "psd_welch" in ops
        assert "reynolds_stats" in ops
        assert "eof" in ops

    def test_describe_returns_dict(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        facade = PostProcessFacade()
        info = facade.describe("psd_welch")
        for k in ("name", "category", "description", "params", "returns"):
            assert k in info

    def test_describe_unknown_raises(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        with pytest.raises(KeyError):
            PostProcessFacade().describe("bogus_op")

    def test_run_unknown_raises(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        with pytest.raises(KeyError):
            PostProcessFacade().run("bogus_op")

    def test_run_invalid_kwargs_raises(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        with pytest.raises(ValueError, match="invalid parameters"):
            PostProcessFacade().run("psd_welch", bogus_kwarg=1)


class TestFacadeOps:
    """모든 op 실행 smoke test — 각 op가 적어도 한 번 동작."""

    def test_psd_welch(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000)
        r = PostProcessFacade().run("psd_welch", signal=x, fs=100.0, nperseg=128)
        assert r["frequency"].shape == r["psd"].shape

    def test_reynolds_stats(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(1)
        u = rng.standard_normal((100, 5))
        r = PostProcessFacade().run("reynolds_stats", u=u)
        assert r["mean"].shape == (5,)
        assert r["rms"].shape == (5,)

    def test_reynolds_stats_3d(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(2)
        u = rng.standard_normal((100, 4))
        v = rng.standard_normal((100, 4))
        w = rng.standard_normal((100, 4))
        r = PostProcessFacade().run("reynolds_stats", u=u, v=v, w=w)
        assert "tke" in r and "intensity" in r

    def test_quadrant_analysis(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(3)
        up = rng.standard_normal(2000)
        vp = rng.standard_normal(2000)
        r = PostProcessFacade().run("quadrant_analysis", up=up, vp=vp)
        assert "Q1" in r["quadrants"]

    def test_kolmogorov_slope(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(4)
        x = rng.standard_normal(512)
        r = PostProcessFacade().run("kolmogorov_slope", signal=x, dx=1.0)
        assert "slope" in r and "r2" in r

    def test_box_stats(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(5)
        x = rng.standard_normal(500)
        r = PostProcessFacade().run("box_stats", x=x)
        assert "median" in r["box"]

    def test_anomaly_mahalanobis(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(6)
        X = rng.standard_normal((100, 3))
        r = PostProcessFacade().run("anomaly_mahalanobis", X=X)
        assert r["scores"].shape == (100,)

    def test_ts_features(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(7)
        x = rng.standard_normal(200)
        r = PostProcessFacade().run("ts_features", signal=x)
        assert "mean" in r["features"]

    def test_change_points_binary(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(8)
        x = np.concatenate([rng.standard_normal(50), 5 + rng.standard_normal(50)])
        r = PostProcessFacade().run(
            "change_points", signal=x, n_changepoints=1, method="binary",
        )
        assert "changepoints" in r

    def test_change_points_pelt(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(9)
        x = np.concatenate([rng.standard_normal(80), 5 + rng.standard_normal(80)])
        r = PostProcessFacade().run("change_points", signal=x, method="pelt")
        assert isinstance(r["changepoints"], list)

    def test_change_points_invalid_method(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        with pytest.raises((ValueError, RuntimeError)):
            PostProcessFacade().run(
                "change_points", signal=np.zeros(100), method="bogus",
            )

    def test_denoise(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(10)
        x = np.sin(np.linspace(0, 4 * np.pi, 200)) + 0.3 * rng.standard_normal(200)
        r = PostProcessFacade().run(
            "denoise", signal=x, window_length=21, polyorder=3,
        )
        assert r["smoothed"].shape == x.shape

    def test_phase_average(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        t = np.linspace(0, 50, 2000)
        x = np.sin(2 * np.pi * t)
        r = PostProcessFacade().run(
            "phase_average", t=t, signal=x, period=1.0, n_bins=20,
        )
        assert r["mean"].shape == (20,)

    def test_eof(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(11)
        X = rng.standard_normal((100, 30))
        r = PostProcessFacade().run("eof", X=X, n_modes=5)
        assert r["eofs"].shape == (30, 5)

    def test_safe_eval(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        u = np.array([1.0, 2.0, 3.0])
        v = np.array([0.0, 1.0, 0.0])
        r = PostProcessFacade().run(
            "safe_eval", expression="sqrt(u**2 + v**2)",
            variables={"u": u, "v": v},
        )
        np.testing.assert_allclose(r["result"], [1.0, np.sqrt(5), 3.0])

    def test_two_point_acf(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(12)
        u = rng.standard_normal((100, 50))
        r = PostProcessFacade().run("two_point_acf", u=u, dx=0.1)
        assert r["R"][0] == 1.0
        assert "L_int" in r

    def test_running_moments(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        rng = np.random.default_rng(13)
        samples = rng.standard_normal((100, 4))
        r = PostProcessFacade().run("running_moments", samples=samples)
        assert r["n"] == 100

    def test_pod_truncation(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        s = np.array([10.0, 5.0, 1.0, 0.1])
        r = PostProcessFacade().run(
            "pod_truncation", singular_values=s, fraction=0.99,
        )
        assert r["n_modes"] >= 1
        assert r["cumulative_energy"].shape == (4,)

    def test_quantile(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        x = np.arange(101.0)
        r = PostProcessFacade().run("quantile", x=x, q=50.0)
        np.testing.assert_allclose(r["value"], 50.0)

    def test_critical_points(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        x = np.linspace(-1, 1, 41)
        X, Y = np.meshgrid(x, x, indexing="ij")
        u, v = -Y, X
        r = PostProcessFacade().run(
            "critical_points", u=u, v=v, dx=0.1, dy=0.1,
        )
        assert r["count"] >= 1


class TestAllOpsCallable:
    """모든 등록된 op이 list_operations에서 노출되고 describe()로 조회 가능."""

    def test_all_ops_describable(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        facade = PostProcessFacade()
        for op_name in facade.list_operations():
            info = facade.describe(op_name)
            assert info["name"] == op_name
            assert info["category"]
            assert info["description"]
            assert isinstance(info["params"], list)
            assert isinstance(info["returns"], list)
