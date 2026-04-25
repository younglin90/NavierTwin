"""Cyclone separator — Lapple cut diameter d50, fractional efficiency.

Examples:
    >>> from naviertwin.core.applied.cyclone import lapple_d50, fraction_efficiency
    >>> d50 = lapple_d50(W=0.1, mu=1.8e-5, Ne=5, Vi=15, rho_p=2000, rho_g=1.2)
    >>> 0 < fraction_efficiency(dp=d50, d50=d50) <= 1
    True
"""

from __future__ import annotations

import math


def lapple_d50(
    *, W: float, mu: float, Ne: float, Vi: float,
    rho_p: float, rho_g: float,
) -> float:
    """Cut diameter d50 (m). W = inlet width, Vi = inlet vel."""
    return float(math.sqrt(9 * mu * W / (2 * math.pi * Ne * Vi * (rho_p - rho_g))))


def fraction_efficiency(*, dp: float, d50: float) -> float:
    """η_d = 1 / (1 + (d50/dp)²)."""
    return float(1.0 / (1.0 + (d50 / max(dp, 1e-30)) ** 2))


__all__ = ["fraction_efficiency", "lapple_d50"]
