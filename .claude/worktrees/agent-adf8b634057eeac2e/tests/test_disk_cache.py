"""Round 245 — disk cache."""

from __future__ import annotations

from pathlib import Path


class TestDC:
    def test_caches(self, tmp_path: Path) -> None:
        from naviertwin.utils.disk_cache import disk_cache

        calls = {"n": 0}

        @disk_cache(tmp_path)
        def f(x, y=1):
            calls["n"] += 1
            return x + y

        assert f(2, 3) == 5
        assert f(2, 3) == 5  # cached
        assert calls["n"] == 1

    def test_different_args(self, tmp_path: Path) -> None:
        from naviertwin.utils.disk_cache import disk_cache

        n = [0]

        @disk_cache(tmp_path)
        def f(x):
            n[0] += 1
            return x * 2

        f(1); f(2); f(3); f(1); f(2)  # noqa: E702
        assert n[0] == 3  # 3 unique inputs

    def test_clear(self, tmp_path: Path) -> None:
        from naviertwin.utils.disk_cache import clear_cache, disk_cache

        @disk_cache(tmp_path)
        def f(x):
            return x + 1

        f(1); f(2)  # noqa: E702
        removed = clear_cache(tmp_path)
        assert removed == 2
