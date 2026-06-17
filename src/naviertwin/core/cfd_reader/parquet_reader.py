"""Parquet/Arrow reader — pyarrow optional dependency.

Examples:
    >>> from naviertwin.core.cfd_reader.parquet_reader import has_pyarrow
    >>> isinstance(has_pyarrow(), bool)
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def read_parquet(path: str | Path) -> dict[str, Any]:
    """Parquet → dict[col_name, np.ndarray] (pyarrow 필요)."""
    if not has_pyarrow():
        raise ImportError("pyarrow not installed; pip install pyarrow")
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    columns = map(lambda col: (col, np.asarray(table.column(col).to_pylist())), table.column_names)
    return dict(columns)


def write_parquet(path: str | Path, data: dict[str, Any]) -> None:
    if not has_pyarrow():
        raise ImportError("pyarrow not installed; pip install pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table(data)
    pq.write_table(table, str(path))


__all__ = ["has_pyarrow", "read_parquet", "write_parquet"]
