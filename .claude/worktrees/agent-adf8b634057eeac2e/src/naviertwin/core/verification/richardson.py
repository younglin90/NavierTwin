"""Richardson extrapolation — refine f estimate using two grids.

f_R = f_fine + (f_fine - f_coarse) / (r^p - 1).

Examples:
    >>> from naviertwin.core.verification.richardson import richardson
    >>> richardson(f_fine=1.04, f_coarse=1.16, r=2.0, p=2.0)
    1.0
"""

from __future__ import annotations


def richardson(*, f_fine: float, f_coarse: float, r: float = 2.0, p: float = 2.0) -> float:
    return float(f_fine + (f_fine - f_coarse) / (r ** p - 1.0))


__all__ = ["richardson"]
