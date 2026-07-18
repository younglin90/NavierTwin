"""Dataset content-addressable storage hash.

Examples:
    >>> from naviertwin.utils.dataset_cas import cas_hash, version_id
    >>> cas_hash(b'data') != cas_hash(b'other')
    True
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def cas_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cas_hash_file(path: str | Path, *, chunk: int = 8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            d = f.read(chunk)
            if not d:
                break
            h.update(d)
    return h.hexdigest()


def version_id(name: str, content_hash: str) -> str:
    return f"{name}@{content_hash[:12]}"


__all__ = ["cas_hash", "cas_hash_file", "version_id"]
