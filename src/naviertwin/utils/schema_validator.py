"""Config schema validator — minimal (no jsonschema dep).

Examples:
    >>> from naviertwin.utils.schema_validator import validate
    >>> ok, errs = validate({"a": 1, "b": "x"}, {"a": int, "b": str})
    >>> ok and errs == []
    True
"""

from __future__ import annotations


def validate(data: dict, schema: dict) -> tuple[bool, list[str]]:
    """schema = {key: type-or-callable}. callable returns bool."""
    errors: list[str] = []
    items = list(schema.items())
    idx = 0
    while idx < len(items):
        key, expected = items[idx]
        if key not in data:
            errors.append(f"missing key: {key}")
            idx += 1
            continue
        v = data[key]
        if isinstance(expected, type):
            if not isinstance(v, expected):
                errors.append(f"{key}: expected {expected.__name__}, got {type(v).__name__}")
        elif callable(expected):
            if not expected(v):
                errors.append(f"{key}: failed predicate")
        else:
            errors.append(f"{key}: invalid schema entry")
        idx += 1
    return (len(errors) == 0), errors


__all__ = ["validate"]
