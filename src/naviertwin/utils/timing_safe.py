"""Timing-safe string compare — wrapper around hmac.compare_digest.

Examples:
    >>> from naviertwin.utils.timing_safe import equal
    >>> equal('abc', 'abc')
    True
    >>> equal('abc', 'abd')
    False
"""

from __future__ import annotations

import hmac


def equal(a: str | bytes, b: str | bytes) -> bool:
    if isinstance(a, str):
        a = a.encode()
    if isinstance(b, str):
        b = b.encode()
    return hmac.compare_digest(a, b)


__all__ = ["equal"]
