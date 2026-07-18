"""Extended PVD writer — group/part annotations.

Examples:
    >>> from pathlib import Path
    >>> import tempfile
    >>> from naviertwin.core.visualization.pvd_extended import write_pvd_grouped
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = Path(d) / 'x.pvd'
    ...     write_pvd_grouped(p, [(0.0, 'fluid', 0, 'f0.vtu'), (0.0, 'solid', 0, 's0.vtu')])
    ...     'group="fluid"' in p.read_text()
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


def _grouped_row(entry: tuple[float, str, int, str]) -> str:
    t, group, part, filename = entry
    return f'    <DataSet timestep="{t}" group="{group}" part="{part}" file="{filename}"/>'


def write_pvd_grouped(
    path: str | Path,
    entries: list[tuple[float, str, int, str]],
) -> None:
    """entries = [(time, group, part, file)]"""
    rows = "\n".join(map(_grouped_row, entries))
    Path(path).write_text(_TEMPLATE.format(rows=rows))


__all__ = ["write_pvd_grouped"]
