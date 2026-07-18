"""Power-law (Ostwald-de Waele) viscosity: μ_app = K |γ̇|^(n-1).

Examples:
    >>> from naviertwin.core.rheology.power_law import apparent_viscosity
    >>> apparent_viscosity(gamma_dot=1.0, K=2.0, n=0.5)
    2.0
"""

from __future__ import annotations

import numpy as np


def apparent_viscosity(*, gamma_dot: float, K: float, n: float) -> float:
    g = max(abs(float(gamma_dot)), 1e-30)
    return float(K * g ** (n - 1))


def shear_stress(*, gamma_dot: float, K: float, n: float) -> float:
    return float(K * np.sign(gamma_dot) * abs(gamma_dot) ** n)


__all__ = ["apparent_viscosity", "shear_stress"]
