"""Variational Multiscale (VMS) — sub-grid stabilization parameter τ.

τ = 1/√((2|u|/h)² + (4ν/h²)²) (Hughes 1995).

Examples:
    >>> from naviertwin.core.multiscale.vms import tau_supg
    >>> tau_supg(u=1.0, h=0.1, nu=0.01)
"""

from __future__ import annotations

import numpy as np


def tau_supg(*, u: float, h: float, nu: float = 0.0) -> float:
    a = (2.0 * abs(u) / h) ** 2
    d = (4.0 * nu / (h * h)) ** 2 if nu > 0 else 0.0
    return float(1.0 / np.sqrt(a + d + 1e-30))


def vms_residual_correction(
    R: np.ndarray, *, tau: float,
) -> np.ndarray:
    """Sub-grid model: u' ≈ -τ R."""
    return -tau * np.asarray(R)


__all__ = ["tau_supg", "vms_residual_correction"]
