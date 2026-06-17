"""Hash-chain audit log — each entry hashes prev hash + msg.

Examples:
    >>> from naviertwin.utils.hash_chain import HashChain
    >>> c = HashChain()
    >>> c.append('event1')
    >>> c.append('event2')
    >>> c.verify()
    True
"""

from __future__ import annotations

import hashlib


class HashChain:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []  # (msg, hash)

    def append(self, msg: str) -> str:
        prev = self.entries[-1][1] if self.entries else ""
        h = hashlib.sha256((prev + msg).encode()).hexdigest()
        self.entries.append((msg, h))
        return h

    def verify(self) -> bool:
        prev = ""
        idx = 0
        while idx < len(self.entries):
            msg, h = self.entries[idx]
            expected = hashlib.sha256((prev + msg).encode()).hexdigest()
            if expected != h:
                return False
            prev = h
            idx += 1
        return True


__all__ = ["HashChain"]
