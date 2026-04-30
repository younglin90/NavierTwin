"""Round 4 — LCS / PGD / entropy_gen / FastAPI 테스트."""

from __future__ import annotations

import numpy as np
import pytest


class TestLCS:
    def test_uniform_flow_ftle_near_zero(self) -> None:
        """균일 흐름은 FTLE ≈ 0 이어야."""
        from naviertwin.core.flow_analysis.vortex.lcs import compute_ftle_2d

        def u_fn(t: float, x, y):
            return np.ones_like(x)

        def v_fn(t: float, x, y):
            return np.zeros_like(y)

        ftle = compute_ftle_2d(u_fn, v_fn, nx=10, ny=10, T=1.0, dt=0.1)
        # 경계 효과 제외 내부 영역에서 FTLE 가 0 에 가까워야
        interior = ftle[2:-2, 2:-2]
        assert float(np.abs(interior).max()) < 0.3

    def test_double_gyre_nontrivial(self) -> None:
        """Double-gyre 같은 비트리비얼 유동은 FTLE > 0 영역이 존재."""
        from naviertwin.core.flow_analysis.vortex.lcs import compute_ftle_2d

        def u_fn(t: float, x, y):
            return -np.sin(np.pi * x) * np.cos(np.pi * y)

        def v_fn(t: float, x, y):
            return np.cos(np.pi * x) * np.sin(np.pi * y)

        ftle = compute_ftle_2d(u_fn, v_fn, nx=16, ny=16, T=1.5, dt=0.05)
        assert ftle.max() > 0.0


class TestPGD3D:
    def test_pgd_reconstruction(self) -> None:
        from naviertwin.core.flow_analysis.modal.pgd import compute_pgd_3d, reconstruct_pgd

        rng = np.random.default_rng(0)
        # rank-2 합성 텐서
        F1 = rng.standard_normal(8)
        G1 = rng.standard_normal(6)
        H1 = rng.standard_normal(5)
        F2 = rng.standard_normal(8)
        G2 = rng.standard_normal(6)
        H2 = rng.standard_normal(5)
        X = np.einsum("i,j,k->ijk", F1, G1, H1) + np.einsum("i,j,k->ijk", F2, G2, H2)

        modes = compute_pgd_3d(X, n_modes=3, max_iter=100)
        assert len(modes) == 3
        X_rec = reconstruct_pgd(modes, X.shape)
        err = float(np.linalg.norm(X - X_rec) / np.linalg.norm(X))
        assert err < 0.1

    def test_wrong_ndim_raises(self) -> None:
        from naviertwin.core.flow_analysis.modal.pgd import compute_pgd_3d

        with pytest.raises(ValueError):
            compute_pgd_3d(np.zeros((4, 4)), n_modes=1)


class TestEntropyGeneration:
    def test_uniform_flow_zero(self) -> None:
        from naviertwin.core.flow_analysis.thermofluids.entropy_gen import (
            entropy_generation_2d,
        )

        u = np.ones((10, 10))
        v = np.zeros((10, 10))
        T = 300.0 * np.ones((10, 10))
        s = entropy_generation_2d(u, v, T, dx=0.1, dy=0.1, mu=1e-3, k=0.026)
        # 모든 구배 0 → s_gen ≈ 0
        assert float(np.abs(s).max()) < 1e-10

    def test_shear_flow_positive(self) -> None:
        from naviertwin.core.flow_analysis.thermofluids.entropy_gen import (
            entropy_generation_2d,
        )

        # 간단 전단 u = y
        y = np.linspace(0, 1, 15)
        u = np.tile(y[:, None], (1, 15))
        v = np.zeros((15, 15))
        T = 300.0 * np.ones((15, 15))
        s = entropy_generation_2d(u, v, T, dx=1.0 / 14, dy=1.0 / 14, mu=1e-3, k=0.026)
        assert float(s.mean()) > 0

    def test_negative_T_raises(self) -> None:
        from naviertwin.core.flow_analysis.thermofluids.entropy_gen import (
            entropy_generation_2d,
        )

        with pytest.raises(ValueError):
            entropy_generation_2d(
                np.zeros((3, 3)), np.zeros((3, 3)),
                -np.ones((3, 3)), dx=1, dy=1, mu=1e-3, k=1.0,
            )


class TestFastAPI:
    def test_create_app_and_endpoints(self) -> None:
        fastapi = pytest.importorskip("fastapi")
        HTTPException = fastapi.HTTPException
        from naviertwin import __version__
        from naviertwin.api.server import CouetteReq, PODReq, create_app

        app = create_app()
        assert app.version == __version__
        route_map = {
            route.path: route.endpoint
            for route in app.routes
            if hasattr(route, "path") and hasattr(route, "endpoint")
        }

        # health
        health = route_map["/health"]()
        assert health["status"] == "ok"
        doctor = route_map["/doctor"]()
        assert doctor["status"] in {"ok", "warn", "error"}
        assert any(check["name"] == "python_version" for check in doctor["checks"])

        # 해석해
        couette = route_map["/analytic/couette"](
            CouetteReq(U_top=1.0, H=1.0, n_points=10)
        )
        assert len(couette["velocity"]) == 10
        assert couette["velocity"][-1] == pytest.approx(1.0, rel=1e-6)

        # POD
        X = np.random.default_rng(0).standard_normal((20, 15)).tolist()
        pod = route_map["/reduce/pod"](PODReq(snapshots=X, n_modes=3))
        assert pod["n_modes"] == 3

        # Generic reduce endpoint (incremental_pod)
        inc = route_map["/reduce"](
            PODReq(snapshots=X, n_modes=3, reducer_kind="incremental_pod")
        )
        assert inc["reducer_kind"] == "incremental_pod"
        assert inc["n_modes"] == 3

        # Generic reduce endpoint (mrpod)
        mr = route_map["/reduce"](
            PODReq(snapshots=X, n_modes=2, reducer_kind="mrpod")
        )
        assert mr["reducer_kind"] == "mrpod"
        assert mr["n_modes"] == 6  # 3 scales × 2 modes/scale

        with pytest.raises(HTTPException, match="unknown reducer_kind"):
            route_map["/reduce"](
                PODReq(snapshots=X, n_modes=2, reducer_kind="unknown")
            )
        assert "/simulate/lbm_cavity" in route_map
