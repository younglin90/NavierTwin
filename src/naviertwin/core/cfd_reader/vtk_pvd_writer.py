"""VTK PVD (ParaView collection) — time-series .vtu 모음 XML 작성.

Examples:
    >>> from pathlib import Path
    >>> import tempfile
    >>> from naviertwin.core.cfd_reader.vtk_pvd_writer import write_pvd
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = Path(d) / "x.pvd"
    ...     write_pvd(p, [(0.0, "f0.vtu"), (1.0, "f1.vtu")])
    ...     'file="f0.vtu"' in p.read_text()
    True
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
  <Collection>
{rows}
  </Collection>
</VTKFile>
"""


def _dataset_row(entry: tuple[float, str]) -> str:
    t, filename = entry
    return f'    <DataSet timestep="{t}" group="" part="0" file="{filename}"/>'


def write_pvd(path: str | Path, entries: list[tuple[float, str]]) -> None:
    """entries = [(time, filename), ...]"""
    rows = "\n".join(map(_dataset_row, entries))
    Path(path).write_text(_TEMPLATE.format(rows=rows))


__all__ = ["write_pvd"]
