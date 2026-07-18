"""SU2 restart file writer — ASCII (CSV-like).

Examples:
    >>> import numpy as np, tempfile, pathlib
    >>> from naviertwin.core.cfd_reader.su2_writer import write_restart
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = pathlib.Path(d) / "restart.dat"
    ...     write_restart(p, ["x", "rho"], np.zeros((3, 2)))
    ...     "rho" in p.read_text()
    True
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def write_restart(
    path: str | Path,
    columns: list[str],
    data: NDArray[np.float64],
) -> None:
    """SU2 ASCII restart format: header line + rows."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape[1] != len(columns):
        raise ValueError(f"data ncols={arr.shape[1]} != columns={len(columns)}")
    header_parts = []
    col_idx = 0
    while col_idx < len(columns):
        header_parts.append(f'"{columns[col_idx]}"')
        col_idx += 1
    header = "\t".join(header_parts)
    lines = [header]
    row_idx = 0
    while row_idx < len(arr):
        row = arr[row_idx]
        values = []
        value_idx = 0
        while value_idx < len(row):
            values.append(f"{row[value_idx]:.16e}")
            value_idx += 1
        lines.append("\t".join(values))
        row_idx += 1
    Path(path).write_text("\n".join(lines) + "\n")


def read_restart(path: str | Path) -> tuple[list[str], NDArray[np.float64]]:
    text = Path(path).read_text().strip().splitlines()
    cols = []
    raw_cols = text[0].split("\t")
    col_idx = 0
    while col_idx < len(raw_cols):
        cols.append(raw_cols[col_idx].strip().strip('"'))
        col_idx += 1

    rows = []
    line_idx = 1
    while line_idx < len(text):
        parts = text[line_idx].split("\t")
        row = []
        part_idx = 0
        while part_idx < len(parts):
            row.append(float(parts[part_idx]))
            part_idx += 1
        rows.append(row)
        line_idx += 1
    data = np.array(rows)
    return cols, data


__all__ = ["read_restart", "write_restart"]
