"""Round 313 — async loader."""

from __future__ import annotations


class TestAsyncLoader:
    def test_yields_in_order(self) -> None:
        from naviertwin.core.io.async_loader import AsyncLoader

        loader = AsyncLoader(iter(range(10)), max_buffer=3)
        result = list(loader.iter())
        assert result == list(range(10))

    def test_propagates_error(self) -> None:
        import pytest

        from naviertwin.core.io.async_loader import AsyncLoader

        def gen():
            yield 1
            yield 2
            raise RuntimeError("boom")

        loader = AsyncLoader(gen(), max_buffer=2)
        with pytest.raises(RuntimeError):
            list(loader.iter())
