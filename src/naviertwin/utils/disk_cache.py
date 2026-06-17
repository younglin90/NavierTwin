"""디스크 기반 함수 결과 캐시 — 인자 해시 기반.

Examples:
    >>> # test 참조
"""

from __future__ import annotations

import pickle
from functools import wraps
from pathlib import Path
from typing import Callable

from naviertwin.utils.hashing import hash_bytes


def disk_cache(cache_dir: str | Path):
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    def deco(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = pickle.dumps((fn.__qualname__, args, sorted(kwargs.items())))
            h = hash_bytes(key)
            path = root / f"{h}.pkl"
            if path.exists():
                with path.open("rb") as f:
                    return pickle.load(f)
            result = fn(*args, **kwargs)
            with path.open("wb") as f:
                pickle.dump(result, f)
            return result
        return wrapper
    return deco


def clear_cache(cache_dir: str | Path) -> int:
    """반환: 제거된 파일 수."""
    root = Path(cache_dir)
    if not root.exists():
        return 0
    n = 0
    paths = list(root.glob("*.pkl"))
    idx = 0
    while idx < len(paths):
        p = paths[idx]
        p.unlink()
        n += 1
        idx += 1
    return n


__all__ = ["disk_cache", "clear_cache"]
