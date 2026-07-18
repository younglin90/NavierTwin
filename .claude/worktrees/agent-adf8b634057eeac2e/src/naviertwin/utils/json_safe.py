"""NumPy / Path / dataclass 안전 JSON 직렬화.

json.dumps 에 cls=NumpySafeEncoder, 또는 safe_dumps/safe_loads 직접 호출.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.json_safe import safe_dumps, safe_loads
    >>> s = safe_dumps({"a": np.array([1, 2, 3]), "b": np.float64(1.5)})
    >>> "1.5" in s
    True
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpySafeEncoder(json.JSONEncoder):
    """NumPy scalar/array + Path + dataclass 지원 encoder."""

    def default(self, obj: Any) -> Any:  # noqa: ARG002
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        try:
            return super().default(obj)
        except TypeError:
            return repr(obj)


def safe_dumps(
    obj: Any, *, indent: int | None = 2, ensure_ascii: bool = False, **kw: Any
) -> str:
    return json.dumps(
        obj, cls=NumpySafeEncoder, indent=indent,
        ensure_ascii=ensure_ascii, **kw,
    )


def safe_loads(s: str) -> Any:
    return json.loads(s)


def safe_dump_file(obj: Any, path: str | Path, **kw: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(safe_dumps(obj, **kw), encoding="utf-8")
    return p


__all__ = ["NumpySafeEncoder", "safe_dumps", "safe_loads", "safe_dump_file"]
