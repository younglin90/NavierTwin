"""Low-Mach preconditioning — Weiss & Smith 1995, eigenvalue rescaling.

c_p² = ε² c²,  ε = min(1, M_local / M_cut).

Examples:
    >>> from naviertwin.core.solvers.low_mach import precond_speed
    >>> precond_speed(M_local=0.01, M_cut=0.3, c=340.0) < 340.0
    True
"""

from __future__ import annotations

import numpy as np


def precond_factor(M_local: float, *, M_cut: float = 0.3) -> float:
    """ε = min(1, M_local / M_cut)."""
    return float(min(1.0, abs(M_local) / max(M_cut, 1e-30)))


def precond_speed(
    *, M_local: float, M_cut: float = 0.3, c: float = 340.0,
) -> float:
    return precond_factor(M_local, M_cut=M_cut) * c


def precond_eigvals(
    u: float, c: float, *, M_local: float, M_cut: float = 0.3,
) -> np.ndarray:
    """3 eigvals of preconditioned 1D Euler."""
    eps = precond_factor(M_local, M_cut=M_cut)
    return np.array([
        u * (1 - eps * eps) / (1 + eps * eps) - eps * c / (1 + eps * eps),
        u,
        u * (1 - eps * eps) / (1 + eps * eps) + eps * c / (1 + eps * eps),
    ])


__all__ = ["precond_eigvals", "precond_factor", "precond_speed"]
