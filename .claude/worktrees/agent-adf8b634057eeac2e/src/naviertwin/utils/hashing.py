"""파일/배열 해싱 — 캐시 fingerprint + 중복 검사.

Examples:
    >>> from pathlib import Path
    >>> from naviertwin.utils.hashing import hash_bytes
    >>> hash_bytes(b"hello")[:6]
    '2cf24d'
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def hash_bytes(data: bytes, algo: str = "sha256") -> str:
    return hashlib.new(algo, data).hexdigest()


def hash_file(path: str | Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_array(arr: np.ndarray, algo: str = "sha256") -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.new(algo)
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def hash_dict(d: dict, algo: str = "sha256") -> str:
    import json
    s = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.new(algo, s).hexdigest()


__all__ = ["hash_bytes", "hash_file", "hash_array", "hash_dict"]
