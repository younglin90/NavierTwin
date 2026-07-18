"""TOML loader wrapper — Python 3.11+ tomllib 사용.

Examples:
    >>> from naviertwin.utils.safe_toml import toml_loads
    >>> toml_loads('x = 1\\ny = "hi"\\n')
    {'x': 1, 'y': 'hi'}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def toml_loads(text: str) -> dict[str, Any]:
    try:
        import tomllib

        return tomllib.loads(text)
    except ImportError:
        try:
            import tomli

            return tomli.loads(text)
        except ImportError as exc:
            raise RuntimeError("tomllib/tomli 필요 (Python >= 3.11)") from exc


def toml_load_file(path: str | Path) -> dict[str, Any]:
    try:
        import tomllib

        with Path(path).open("rb") as f:
            return tomllib.load(f)
    except ImportError:
        return toml_loads(Path(path).read_text(encoding="utf-8"))


def _dump_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return '"' + v.replace('"', '\\"') + '"'
    if isinstance(v, list):
        values = []
        value_idx = 0
        while value_idx < len(v):
            values.append(_dump_value(v[value_idx]))
            value_idx += 1
        return "[" + ", ".join(values) + "]"
    return '"' + str(v) + '"'


def toml_dump(data: dict[str, Any]) -> str:
    """간단 TOML 덤프 — 최상위 key only (표준 TOML 포맷은 아니지만 round-trip)."""
    lines = []
    tables = {}
    items = list(data.items())
    item_idx = 0
    while item_idx < len(items):
        k, v = items[item_idx]
        if isinstance(v, dict):
            tables[k] = v
        else:
            lines.append(f"{k} = {_dump_value(v)}")
        item_idx += 1
    table_items = list(tables.items())
    table_idx = 0
    while table_idx < len(table_items):
        name, tbl = table_items[table_idx]
        lines.append(f"\n[{name}]")
        tbl_items = list(tbl.items())
        tbl_idx = 0
        while tbl_idx < len(tbl_items):
            k, v = tbl_items[tbl_idx]
            lines.append(f"{k} = {_dump_value(v)}")
            tbl_idx += 1
        table_idx += 1
    return "\n".join(lines) + "\n"


__all__ = ["toml_loads", "toml_load_file", "toml_dump"]
