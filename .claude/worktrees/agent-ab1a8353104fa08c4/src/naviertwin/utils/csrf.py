"""CSRF token — generate + verify with HMAC.

Examples:
    >>> from naviertwin.utils.csrf import generate_token, verify_token
    >>> tok = generate_token('session1', secret='k')
    >>> verify_token('session1', tok, secret='k')
    True
"""

from __future__ import annotations

import hashlib
import hmac
import secrets


def generate_token(session_id: str, *, secret: str) -> str:
    nonce = secrets.token_hex(8)
    sig = hmac.new(
        secret.encode(), (session_id + nonce).encode(), hashlib.sha256,
    ).hexdigest()
    return f"{nonce}.{sig}"


def verify_token(session_id: str, token: str, *, secret: str) -> bool:
    parts = token.split(".")
    if len(parts) != 2:
        return False
    nonce, sig = parts
    expected = hmac.new(
        secret.encode(), (session_id + nonce).encode(), hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, sig)


__all__ = ["generate_token", "verify_token"]
