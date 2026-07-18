"""OpenFOAM result writer — internalField scalar/vector ASCII.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.cfd_reader.foam_writer import write_internal_scalar
    >>> import tempfile, pathlib
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = pathlib.Path(d) / "p"
    ...     write_internal_scalar(p, "p", np.arange(3.0))
    ...     "internalField" in p.read_text()
    True
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def _scalar_line(x: float) -> str:
    return f"{x}\n"


def _vector_line(row: NDArray[np.float64]) -> str:
    a, b, c = row
    return f"({a} {b} {c})\n"


def write_internal_scalar(
    path: str | Path,
    name: str,
    values: NDArray[np.float64],
) -> None:
    v = np.asarray(values, dtype=np.float64).ravel()
    lines = [_HEADER.format(cls="volScalarField", obj=name)]
    lines.append("dimensions      [0 0 0 0 0 0 0];\n")
    lines.append(f"internalField   nonuniform List<scalar>\n{len(v)}\n(\n")
    lines.extend(map(_scalar_line, v))
    lines.append(");\n")
    lines.append("boundaryField {}\n")
    Path(path).write_text("".join(lines))


def write_internal_vector(
    path: str | Path,
    name: str,
    values: NDArray[np.float64],
) -> None:
    v = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    lines = [_HEADER.format(cls="volVectorField", obj=name)]
    lines.append("dimensions      [0 1 -1 0 0 0 0];\n")
    lines.append(f"internalField   nonuniform List<vector>\n{len(v)}\n(\n")
    lines.extend(map(_vector_line, v))
    lines.append(");\n")
    lines.append("boundaryField {}\n")
    Path(path).write_text("".join(lines))


__all__ = ["write_internal_scalar", "write_internal_vector"]
