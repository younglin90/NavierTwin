"""실행 시간 측정 — 데코레이터 / context manager / 누적 timer.

Examples:
    >>> from naviertwin.utils.timing import timer, Stopwatch
    >>> @timer
    ... def slow():
    ...     sum(range(100000))
    >>> slow()  # doctest: +ELLIPSIS
    >>> sw = Stopwatch()
    >>> with sw:
    ...     sum(range(100))
    >>> sw.elapsed >= 0
    True
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, TypeVar

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def timer(fn: Callable[..., T]) -> Callable[..., T]:
    """함수 실행 시간을 logger 에 기록."""
    @wraps(fn)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            logger.info("⏱ %s: %.4fs", fn.__name__, dt)
    return wrapper


@contextmanager
def timed(name: str = "block"):
    """with timed('loop'): ... → logger 출력."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("⏱ %s: %.4fs", name, time.perf_counter() - t0)


class Stopwatch:
    """시작/정지/누적/재설정."""

    def __init__(self) -> None:
        self._start: float | None = None
        self._accum: float = 0.0
        self._running: bool = False

    def start(self) -> "Stopwatch":
        if not self._running:
            self._start = time.perf_counter()
            self._running = True
        return self

    def stop(self) -> float:
        if self._running and self._start is not None:
            self._accum += time.perf_counter() - self._start
            self._running = False
            self._start = None
        return self._accum

    def reset(self) -> None:
        self._accum = 0.0
        self._running = False
        self._start = None

    @property
    def elapsed(self) -> float:
        base = self._accum
        if self._running and self._start is not None:
            base += time.perf_counter() - self._start
        return base

    def __enter__(self) -> "Stopwatch":
        return self.start()

    def __exit__(self, *exc) -> None:  # noqa: ANN001
        self.stop()


__all__ = ["timer", "timed", "Stopwatch"]
