"""Round 25 — DiskCache + API LBM 엔드포인트 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestDiskCache:
    def test_get_or_compute(self, tmp_path: Path) -> None:
        from naviertwin.utils.cache import DiskCache

        cache = DiskCache(tmp_path)
        call_count = {"n": 0}

        def compute() -> int:
            call_count["n"] += 1
            return 42

        v1 = cache.get_or_compute("key1", compute)
        v2 = cache.get_or_compute("key1", compute)
        assert v1 == 42 and v2 == 42
        # 두번째 호출은 compute 실행 안됨
        assert call_count["n"] == 1

    def test_put_get_has(self, tmp_path: Path) -> None:
        from naviertwin.utils.cache import DiskCache

        c = DiskCache(tmp_path)
        assert not c.has("x")
        c.put("x", {"a": 1})
        assert c.has("x")
        assert c.get("x") == {"a": 1}

    def test_clear(self, tmp_path: Path) -> None:
        from naviertwin.utils.cache import DiskCache

        c = DiskCache(tmp_path)
        c.put("a", 1)
        c.put("b", 2)
        c.clear()
        assert not c.has("a") and not c.has("b")


class TestAPILBMEndpoint:
    def test_lbm_cavity(self) -> None:
        fastapi = pytest.importorskip("fastapi")
        del fastapi

        from naviertwin.api import LBMReq, create_app

        app = create_app()
        route_map = {
            route.path: route.endpoint
            for route in app.routes
            if hasattr(route, "path") and hasattr(route, "endpoint")
        }
        body = route_map["/simulate/lbm_cavity"](
            LBMReq(nx=6, ny=6, tau=0.8, u_top=0.05, n_steps=2, record_every=1)
        )
        assert body["n_snapshots"] == 2
        assert body["shape"] == [2, 6, 6, 3]
        assert abs(body["ux_max"]) < 1.0
