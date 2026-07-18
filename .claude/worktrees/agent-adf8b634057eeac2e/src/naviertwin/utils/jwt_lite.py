"""JWT-lite — HS256 only, no kid/alg negotiation.

Examples:
    >>> from naviertwin.utils.jwt_lite import encode_jwt, decode_jwt
    >>> tok = encode_jwt({'sub': 'alice'}, secret='k')
    >>> decode_jwt(tok, secret='k')
    {'sub': 'alice'}
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def encode_jwt(claims: dict[str, Any], *, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url_encode(json.dumps(claims, separators=(",", ":")).encode())
    sig = hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url_encode(sig)}"


def decode_jwt(token: str, *, secret: str) -> dict[str, Any]:
    h, p, s = token.split(".")
    expected = hmac.new(
        secret.encode(), f"{h}.{p}".encode(), hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(expected, _b64url_decode(s)):
        raise ValueError("invalid signature")
    return json.loads(_b64url_decode(p))


__all__ = ["decode_jwt", "encode_jwt"]
