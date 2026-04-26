"""Ghia, Ghia & Shin (1982) lid-driven cavity reference data.

Tabulated centerline velocity profiles from
"High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations
and a Multigrid Method", J. Comput. Phys. 48, 387-411 (1982).

Only Re = 100 is bundled (public, single table excerpt) — sufficient for V&V
of the in-tree projection solver. For higher Re, fetch the full paper data
into a separate dataset.

Examples:
    >>> from naviertwin.core.benchmarks.ghia_cavity import ghia_u_centerline
    >>> y, u = ghia_u_centerline(Re=100)
    >>> len(y) == len(u)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# y / L,  u / U_lid  along the vertical centerline (x=0.5) for Re=100
_GHIA_RE100 = np.array([
    [1.0000, 1.00000],
    [0.9766, 0.84123],
    [0.9688, 0.78871],
    [0.9609, 0.73722],
    [0.9531, 0.68717],
    [0.8516, 0.23151],
    [0.7344, 0.00332],
    [0.6172, -0.13641],
    [0.5000, -0.20581],
    [0.4531, -0.21090],
    [0.2813, -0.15662],
    [0.1719, -0.10150],
    [0.1016, -0.06434],
    [0.0703, -0.04775],
    [0.0625, -0.04192],
    [0.0547, -0.03717],
    [0.0000, 0.00000],
])


def ghia_u_centerline(Re: float = 100) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Returns (y/L, u/U_lid) for the vertical-centerline u-profile."""
    if int(Re) != 100:
        raise ValueError(f"Only Re=100 bundled, got {Re}")
    return _GHIA_RE100[:, 0].copy(), _GHIA_RE100[:, 1].copy()


__all__ = ["ghia_u_centerline"]
