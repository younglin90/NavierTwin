"""Taylor-Green vortex 해석해 — 2D 비압축 NS 검증에 표준적.

2D: u(x,y,t) = sin(x)cos(y) exp(-2ν t)
    v(x,y,t) = -cos(x)sin(y) exp(-2ν t)
    p(x,y,t) = -¼ρ(cos2x+cos2y) exp(-4ν t)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.validation.taylor_green import taylor_green_2d
    >>> x = np.linspace(0, 2*np.pi, 8)
    >>> y = np.linspace(0, 2*np.pi, 8)
    >>> res = taylor_green_2d(x, y, t=0.0, nu=0.01)
    >>> set(res.keys()) >= {"u", "v", "p"}
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def taylor_green_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    t: float = 0.0,
    nu: float = 1e-2,
    rho: float = 1.0,
) -> dict[str, NDArray[np.float64]]:
    """2D Taylor-Green (meshgrid indexing='xy')."""
    X, Y = np.meshgrid(x, y, indexing="xy")
    decay = np.exp(-2.0 * nu * t)
    u = np.sin(X) * np.cos(Y) * decay
    v = -np.cos(X) * np.sin(Y) * decay
    p = -0.25 * rho * (np.cos(2 * X) + np.cos(2 * Y)) * (decay ** 2)
    return {"u": u, "v": v, "p": p, "X": X, "Y": Y}


def kinetic_energy_decay(t: float, nu: float) -> float:
    """TG vortex 의 kinetic energy 시간 감쇠: E(t) = E(0) e^{-4νt}."""
    return float(np.exp(-4.0 * nu * t))


__all__ = ["taylor_green_2d", "kinetic_energy_decay"]
