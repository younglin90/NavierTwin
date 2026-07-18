"""스레드 기반 함수 타임아웃 — 장시간 블로킹 방지.

Examples:
    >>> from naviertwin.utils.watchdog import run_with_timeout
"""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class TimeoutError_(Exception):
    """내부 타임아웃 예외 (내장과 혼동 방지)."""


def run_with_timeout(
    fn: Callable[..., T], timeout: float, *args: Any, **kwargs: Any,
) -> T:
    """timeout (초) 초과 시 TimeoutError 발생."""
    result: list[Any] = [None]
    err: list[BaseException | None] = [None]

    def worker():
        try:
            result[0] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001
            err[0] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError_(f"fn {fn.__name__} timed out after {timeout}s")
    if err[0] is not None:
        raise err[0]
    return result[0]  # type: ignore[return-value]


__all__ = ["run_with_timeout", "TimeoutError_"]
