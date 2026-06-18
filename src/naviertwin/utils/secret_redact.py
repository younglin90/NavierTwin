"""Secret redaction — mask common patterns (api keys, tokens) in logs.

Examples:
    >>> from naviertwin.utils.secret_redact import redact
    >>> redact('token=abcdef123456')
    'token=***REDACTED***'
"""

from __future__ import annotations

import re
from typing import Any

_PATTERNS = [
    re.compile(
        r"(token|api[_-]?key|secret|password|passwd|aws_secret_access_key)\s*[=:]\s*[\w\-./]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b[A-Z][A-Z0-9_]*(TOKEN|API[_-]?KEY|SECRET|PASSWORD|PASSWD)\b\s*=\s*[^\s\"']+",
        re.IGNORECASE,
    ),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----"),
]
_SENSITIVE_KEY = re.compile(
    r"(token|api[_-]?key|secret|password|passwd|auth|credential|bearer|session|cookie|jwt)",
    re.IGNORECASE,
)


def redact(text: str) -> str:
    out = text
    idx = 0
    while idx < len(_PATTERNS):
        pat = _PATTERNS[idx]
        out = pat.sub(lambda m: m.group(0).split("=")[0].split(":")[0].rstrip()
                       + "=***REDACTED***" if "=" in m.group(0) or ":" in m.group(0)
                       else "***REDACTED***", out)
        idx += 1
    return out


def is_sensitive_key(name: str) -> bool:
    """Return whether a key name suggests sensitive data."""
    return bool(_SENSITIVE_KEY.search(name))


def redact_object(value: Any, key_name: str | None = None) -> Any:
    """Redact secrets from nested JSON-compatible objects."""
    if isinstance(value, str):
        if key_name is not None and is_sensitive_key(key_name):
            return "***REDACTED***"
        return redact(value)
    if isinstance(value, list):
        out: list[Any] = []
        idx = 0
        while idx < len(value):
            out.append(redact_object(value[idx], key_name=key_name))
            idx += 1
        return out
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        items = list(value.items())
        idx = 0
        while idx < len(items):
            key, item = items[idx]
            child_key = str(key)
            if is_sensitive_key(child_key):
                sanitized[child_key] = "***REDACTED***"
                idx += 1
                continue
            sanitized[child_key] = redact_object(item, key_name=child_key)
            idx += 1
        return sanitized
    return value


__all__ = ["redact", "is_sensitive_key", "redact_object"]
