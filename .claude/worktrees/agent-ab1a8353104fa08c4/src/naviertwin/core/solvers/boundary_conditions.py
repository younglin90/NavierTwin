"""경계조건 모델링 — Dirichlet / Neumann / Robin / Periodic.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.boundary_conditions import BoundaryCondition, apply_bc
    >>> bc = BoundaryCondition(kind="dirichlet", value=0.0)
    >>> u = np.array([5.0, 1.0, 2.0, 3.0, 5.0])
    >>> apply_bc(u, left=bc, right=bc)
    >>> u[0], u[-1]
    (np.float64(0.0), np.float64(0.0))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundaryCondition:
    """경계조건.

    kind: "dirichlet" (u=value), "neumann" (∂u/∂n=value),
          "robin" (αu + β ∂u/∂n = value), "periodic".
    value: 스칼라 또는 Callable(t) → 스칼라.
    alpha, beta: Robin 계수 (α u + β du/dn = value).
    """

    kind: str = "dirichlet"
    value: float | Callable[[float], float] = 0.0
    alpha: float = 1.0
    beta: float = 0.0

    def resolve(self, t: float = 0.0) -> float:
        if callable(self.value):
            return float(self.value(t))
        return float(self.value)


def apply_bc(
    u: NDArray[np.float64],
    *,
    left: BoundaryCondition | None = None,
    right: BoundaryCondition | None = None,
    dx: float = 1.0,
    t: float = 0.0,
) -> None:
    """1D 배열 u 의 양 끝에 경계조건 적용 (in-place)."""
    if left is not None:
        _apply_one(u, left, end="left", dx=dx, t=t)
    if right is not None:
        _apply_one(u, right, end="right", dx=dx, t=t)


def _apply_one(
    u: NDArray[np.float64], bc: BoundaryCondition, *,
    end: str, dx: float, t: float,
) -> None:
    v = bc.resolve(t)
    idx, neighbor_idx, sign = (0, 1, 1.0) if end == "left" else (-1, -2, -1.0)
    if bc.kind == "dirichlet":
        u[idx] = v
    elif bc.kind == "neumann":
        # one-sided FD: (u_boundary - u_neighbor) / (sign * dx) = v
        u[idx] = u[neighbor_idx] + sign * dx * v
    elif bc.kind == "robin":
        # α u + β (u_b - u_n) / (sign dx) = v  →  u_b
        if bc.alpha == 0 and bc.beta == 0:
            raise ValueError("Robin α and β both zero")
        u[idx] = (v - bc.beta * (-sign) * u[neighbor_idx] / dx) / (
            bc.alpha + bc.beta / (sign * dx)
        )
    elif bc.kind == "periodic":
        # 반대쪽에서 가져옴
        u[idx] = u[-2] if end == "left" else u[1]
    else:
        raise ValueError(f"unknown BC kind: {bc.kind}")


def make_bc(kind: str, value: float = 0.0, **kw) -> BoundaryCondition:
    return BoundaryCondition(kind=kind, value=value, **kw)


__all__ = ["BoundaryCondition", "apply_bc", "make_bc"]
