"""결과 캐싱 유틸 — 함수 호출 결과를 디스크에 pickle.

Use cases:
    - 무거운 LBM 시뮬레이션 결과
    - 긴 PINN 학습 체크포인트
    - POD 모드 재사용

Examples:
    >>> from naviertwin.utils.cache import DiskCache
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as d:
    ...     cache = DiskCache(d)
    ...     def slow_compute(x):
    ...         return x ** 2
    ...     v1 = cache.get_or_compute("square_5", lambda: slow_compute(5))
    ...     v2 = cache.get_or_compute("square_5", lambda: slow_compute(5))
    ...     v1 == v2 == 25
    True
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable, TypeVar

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class DiskCache:
    """간단한 pickle 디스크 캐시."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self.dir / f"{h}.pkl"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> Any:
        p = self._path(key)
        if not p.exists():
            raise KeyError(key)
        with p.open("rb") as f:
            return pickle.load(f)

    def put(self, key: str, value: Any) -> None:
        with self._path(key).open("wb") as f:
            pickle.dump(value, f)

    def get_or_compute(
        self, key: str, compute_fn: Callable[[], T]
    ) -> T:
        """key 가 있으면 반환, 없으면 compute_fn() 실행 후 저장."""
        if self.has(key):
            logger.debug("DiskCache HIT: %s", key)
            return self.get(key)
        logger.debug("DiskCache MISS: %s — compute", key)
        value = compute_fn()
        self.put(key, value)
        return value

    def clear(self) -> None:
        paths = list(self.dir.glob("*.pkl"))
        idx = 0
        while idx < len(paths):
            p = paths[idx]
            p.unlink()
            idx += 1


__all__ = ["DiskCache"]
