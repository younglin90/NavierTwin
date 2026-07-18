"""Signed config — HMAC-SHA256 signature of JSON config.

Examples:
    >>> from naviertwin.utils.signed_config import sign_config, verify_config
    >>> cfg = {'epochs': 10}
    >>> sig = sign_config(cfg, key='secret')
    >>> verify_config(cfg, sig, key='secret')
    True
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def _canonical(cfg: dict) -> bytes:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()


def sign_config(cfg: dict[str, Any], *, key: str) -> str:
    return hmac.new(
        key.encode(), _canonical(cfg), hashlib.sha256,
    ).hexdigest()


def verify_config(cfg: dict[str, Any], signature: str, *, key: str) -> bool:
    expected = sign_config(cfg, key=key)
    return hmac.compare_digest(expected, signature)


__all__ = ["sign_config", "verify_config"]
