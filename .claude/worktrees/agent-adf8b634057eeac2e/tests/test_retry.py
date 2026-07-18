"""Round 85 — 지수 백오프 retry 데코레이터."""

from __future__ import annotations

import pytest


class TestRetry:
    def test_success_after_failures(self) -> None:
        from naviertwin.utils.retry import retry

        calls = {"n": 0}

        @retry(max_attempts=4, delay=0.001, exceptions=(RuntimeError,))
        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("fail")
            return "ok"

        assert flaky() == "ok"
        assert calls["n"] == 3

    def test_re_raises_after_max(self) -> None:
        from naviertwin.utils.retry import retry

        @retry(max_attempts=2, delay=0.001, exceptions=(ValueError,))
        def always_fail() -> None:
            raise ValueError("nope")

        with pytest.raises(ValueError):
            always_fail()

    def test_does_not_catch_other(self) -> None:
        from naviertwin.utils.retry import retry

        @retry(max_attempts=3, delay=0.001, exceptions=(ValueError,))
        def type_err() -> None:
            raise TypeError("unrelated")

        with pytest.raises(TypeError):
            type_err()

    def test_invalid_attempts(self) -> None:
        from naviertwin.utils.retry import retry

        with pytest.raises(ValueError):
            retry(max_attempts=0)
