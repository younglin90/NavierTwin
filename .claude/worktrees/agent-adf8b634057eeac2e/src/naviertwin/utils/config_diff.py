"""Nested dict diff — 두 config 간 변경사항 비교.

Examples:
    >>> from naviertwin.utils.config_diff import diff_configs
    >>> a = {"lr": 1e-3, "model": {"layers": 3}}
    >>> b = {"lr": 5e-4, "model": {"layers": 3, "dropout": 0.1}}
    >>> d = diff_configs(a, b)
    >>> "lr" in d["changed"]
    True
"""

from __future__ import annotations

from typing import Any


def diff_configs(a: dict, b: dict, path: str = "") -> dict[str, dict[str, Any]]:
    changed: dict[str, tuple[Any, Any]] = {}
    added: dict[str, Any] = {}
    removed: dict[str, Any] = {}
    keys = list(a.keys() | b.keys())
    idx = 0
    while idx < len(keys):
        k = keys[idx]
        p = f"{path}.{k}" if path else k
        if k in a and k not in b:
            removed[p] = a[k]
        elif k not in a and k in b:
            added[p] = b[k]
        else:
            va, vb = a[k], b[k]
            if isinstance(va, dict) and isinstance(vb, dict):
                sub = diff_configs(va, vb, p)
                changed.update(sub["changed"])
                added.update(sub["added"])
                removed.update(sub["removed"])
            elif va != vb:
                changed[p] = (va, vb)
        idx += 1
    return {"changed": changed, "added": added, "removed": removed}


def format_diff(d: dict[str, dict]) -> str:
    lines: list[str] = []
    changed_items = list(d["changed"].items())
    idx = 0
    while idx < len(changed_items):
        p, (a, b) = changed_items[idx]
        lines.append(f"~ {p}: {a!r} -> {b!r}")
        idx += 1
    added_items = list(d["added"].items())
    idx = 0
    while idx < len(added_items):
        p, v = added_items[idx]
        lines.append(f"+ {p}: {v!r}")
        idx += 1
    removed_items = list(d["removed"].items())
    idx = 0
    while idx < len(removed_items):
        p, v = removed_items[idx]
        lines.append(f"- {p}: {v!r}")
        idx += 1
    return "\n".join(lines)


__all__ = ["diff_configs", "format_diff"]
