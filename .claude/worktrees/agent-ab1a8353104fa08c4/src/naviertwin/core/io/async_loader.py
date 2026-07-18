"""Async producer-consumer loader — bounded queue + worker thread.

Examples:
    >>> from naviertwin.core.io.async_loader import AsyncLoader
    >>> def gen():
    ...     yield from range(5)
    >>> loader = AsyncLoader(gen(), max_buffer=2)
    >>> sum(loader.iter())
    10
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from queue import Queue
from typing import Any

_SENTINEL = object()


class AsyncLoader:
    def __init__(self, source: Iterable[Any], *, max_buffer: int = 4) -> None:
        self.source = source
        self.q: Queue = Queue(maxsize=max_buffer)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.error: Exception | None = None

    def _run(self) -> None:
        try:
            tuple(map(self.q.put, self.source))
            self.q.put(_SENTINEL)
        except Exception as e:  # noqa: BLE001
            self.error = e
            self.q.put(_SENTINEL)

    def iter(self) -> Iterator[Any]:
        self.thread.start()
        while True:
            item = self.q.get()
            if item is _SENTINEL:
                break
            yield item
        if self.error is not None:
            raise self.error


__all__ = ["AsyncLoader"]
