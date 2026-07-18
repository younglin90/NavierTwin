"""병렬 실행 헬퍼 — thread/process map + error isolation.

Examples:
    >>> from naviertwin.utils.parallel import thread_map
    >>> thread_map(lambda x: x * 2, [1, 2, 3], workers=2)
    [2, 4, 6]
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def thread_map(
    fn: Callable[[T], R], items: Iterable[T],
    *, workers: int = 4,
) -> list[R]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def process_map(
    fn: Callable[[T], R], items: Iterable[T],
    *, workers: int = 2,
) -> list[R]:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def safe_thread_map(
    fn: Callable[[T], R], items: Iterable[T],
    *, workers: int = 4,
) -> list[tuple[bool, R | str]]:
    """(ok, result|error_message) 쌍 반환."""
    items_list = list(items)
    results: list[tuple[bool, R | str]] = []
    result_idx = 0
    while result_idx < len(items_list):
        results.append((False, ""))
        result_idx += 1

    def _wrap(i: int, x: T):
        try:
            return i, True, fn(x)
        except Exception as e:  # noqa: BLE001
            return i, False, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        item_idx = 0
        while item_idx < len(items_list):
            futures.append(ex.submit(_wrap, item_idx, items_list[item_idx]))
            item_idx += 1
        future_idx = 0
        while future_idx < len(futures):
            fut = futures[future_idx]
            idx, ok, payload = fut.result()
            results[idx] = (ok, payload)
            future_idx += 1
    return results


__all__ = ["thread_map", "process_map", "safe_thread_map"]
