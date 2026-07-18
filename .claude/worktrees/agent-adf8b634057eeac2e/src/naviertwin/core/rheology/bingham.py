"""Bingham plastic stress and apparent viscosity helpers.

Examples:
    >>> from naviertwin.core.rheology.bingham import bingham_stress
    >>> bingham_stress(gamma_dot=2.0, tau_y=1.0, mu_p=0.5)
    2.0
"""

from __future__ import annotations

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by Bingham rheology")


def bingham_stress(*, gamma_dot: float, tau_y: float, mu_p: float) -> float:
    return float(_kernels.bingham_stress(float(gamma_dot), float(tau_y), float(mu_p)))


def bingham_apparent_viscosity(*, gamma_dot: float, tau_y: float, mu_p: float,
                                eps: float = 1e-6) -> float:
    return float(
        _kernels.bingham_apparent_viscosity(float(gamma_dot), float(tau_y), float(mu_p), float(eps)),
    )


__all__ = ["bingham_apparent_viscosity", "bingham_stress"]
