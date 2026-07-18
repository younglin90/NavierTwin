"""Grid Convergence Index (Roache 1994).

GCI = Fs · |ε| / (r^p - 1).

Examples:
    >>> from naviertwin.core.verification.gci import gci
    >>> gci(eps=0.01, r=2.0, p=2.0, Fs=1.25)
"""

from __future__ import annotations


def gci(*, eps: float, r: float, p: float, Fs: float = 1.25) -> float:
    """eps = (f_fine - f_coarse)/f_fine; r = grid refinement ratio; p = order."""
    return float(Fs * abs(eps) / (r ** p - 1.0))


def observed_order(
    *, f1: float, f2: float, f3: float, r: float = 2.0,
) -> float:
    """f1 finest, f3 coarsest; p ≈ ln((f3-f2)/(f2-f1))/ln(r)."""
    import math
    num = f3 - f2
    den = f2 - f1
    if abs(den) < 1e-30 or num * den <= 0:
        return float("nan")
    return float(math.log(num / den) / math.log(r))


__all__ = ["gci", "observed_order"]
