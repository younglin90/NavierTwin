"""Round 239 — parallel helpers."""

from __future__ import annotations


class TestParallel:
    def test_thread_map(self) -> None:
        from naviertwin.utils.parallel import thread_map

        assert thread_map(lambda x: x * x, [1, 2, 3, 4], workers=2) == [1, 4, 9, 16]

    def test_safe_thread_map_ok(self) -> None:
        from naviertwin.utils.parallel import safe_thread_map

        res = safe_thread_map(lambda x: x + 1, [1, 2, 3], workers=2)
        assert all(ok for ok, _ in res)
        assert [r for _, r in res] == [2, 3, 4]

    def test_safe_thread_map_error(self) -> None:
        from naviertwin.utils.parallel import safe_thread_map

        def bad(x):
            if x == 2:
                raise ValueError("boom")
            return x

        res = safe_thread_map(bad, [1, 2, 3])
        assert res[0][0] is True
        assert res[1][0] is False and "boom" in res[1][1]
        assert res[2][0] is True
