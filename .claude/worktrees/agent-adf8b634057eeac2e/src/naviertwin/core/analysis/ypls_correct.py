"""Wall function — y+ 자동 보정 (log-law / linear sublayer 분리).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.ypls_correct import wall_velocity
    >>> u_tau, u_plus = wall_velocity(np.array([5.0, 50.0, 500.0]))
    >>> u_plus.shape
    (3,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

KAPPA = 0.41
B = 5.5
Y_PLUS_TRANSITION = 11.06


def u_plus_loglaw(y_plus: NDArray[np.float64]) -> NDArray[np.float64]:
    """u+ = (1/κ) ln(y+) + B."""
    y = np.asarray(y_plus, dtype=np.float64)
    return (1.0 / KAPPA) * np.log(np.maximum(y, 1e-30)) + B


def u_plus_blended(y_plus: NDArray[np.float64]) -> NDArray[np.float64]:
    """y+ < 11.06 → u+ = y+; else log-law."""
    y = np.asarray(y_plus, dtype=np.float64)
    log_part = u_plus_loglaw(y)
    return np.where(y < Y_PLUS_TRANSITION, y, log_part)


def wall_velocity(
    y_plus: NDArray[np.float64], u_tau: float = 1.0,
) -> tuple[float, NDArray[np.float64]]:
    """편의: u_tau, u_plus 반환."""
    return float(u_tau), u_plus_blended(y_plus)


__all__ = [
    "B",
    "KAPPA",
    "Y_PLUS_TRANSITION",
    "u_plus_blended",
    "u_plus_loglaw",
    "wall_velocity",
]
