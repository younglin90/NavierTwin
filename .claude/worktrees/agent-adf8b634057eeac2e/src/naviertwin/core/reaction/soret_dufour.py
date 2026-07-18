"""Soret + Dufour cross-diffusion fluxes.

J_mass = -ρ D ∇Y - ρ D_T Y(1-Y) ∇T/T  (Soret)
q_extra = -k D_dufour ∇Y                (Dufour)

Examples:
    >>> from naviertwin.core.reaction.soret_dufour import soret_flux, dufour_heat
    >>> soret_flux(rho=1, D=1e-5, D_T=1e-7, Y=0.5, gradY=1.0, gradT=10.0, T=300)
"""

from __future__ import annotations


def soret_flux(*, rho: float, D: float, D_T: float, Y: float,
                gradY: float, gradT: float, T: float) -> float:
    return float(-rho * D * gradY - rho * D_T * Y * (1 - Y) * gradT / max(T, 1e-12))


def dufour_heat(*, k_dufour: float, gradY: float) -> float:
    return float(-k_dufour * gradY)


__all__ = ["dufour_heat", "soret_flux"]
