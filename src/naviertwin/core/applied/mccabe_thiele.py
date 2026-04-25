"""McCabe-Thiele single equilibrium stage step.

y* = α x / (1 + (α-1) x).
op line: y = (R/(R+1)) x + x_D/(R+1).

Examples:
    >>> from naviertwin.core.applied.mccabe_thiele import equilibrium_y, op_line_x
    >>> equilibrium_y(x=0.5, alpha=2.0)
    0.6666666666666666
"""

from __future__ import annotations


def equilibrium_y(*, x: float, alpha: float) -> float:
    return float(alpha * x / (1.0 + (alpha - 1.0) * x))


def op_line_x(*, y: float, R: float, x_D: float) -> float:
    """Inverse of y = (R/(R+1)) x + x_D/(R+1)."""
    return float((y - x_D / (R + 1)) * (R + 1) / R)


__all__ = ["equilibrium_y", "op_line_x"]
