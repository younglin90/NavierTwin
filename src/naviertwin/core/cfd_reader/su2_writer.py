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
    header = "\t".join(f'"{c}"' for c in columns)
    lines = [header]
    lines.extend("\t".join(f"{x:.16e}" for x in row) for row in arr)
    Path(path).write_text("\n".join(lines) + "\n")


def read_restart(path: str | Path) -> tuple[list[str], NDArray[np.float64]]:
    text = Path(path).read_text().strip().splitlines()
    cols = [c.strip().strip('"') for c in text[0].split("\t")]
    data = np.array([
        [float(x) for x in line.split("\t")] for line in text[1:]
    ])
    return cols, data


__all__ = ["read_restart", "write_restart"]
