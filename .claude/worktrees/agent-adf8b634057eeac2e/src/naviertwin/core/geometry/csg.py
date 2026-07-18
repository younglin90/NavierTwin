"""CSG primitives via SDF — union/intersect/difference.

Examples:
    >>> from naviertwin.core.geometry.csg import csg_union
    >>> csg_union(0.5, -0.2)
    -0.2
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def csg_union(*phis: Any) -> Any:
    """min: outside both → +; inside either → −."""
    return np.minimum.reduce(list(map(np.asarray, phis)))


def csg_intersect(*phis: Any) -> Any:
    return np.maximum.reduce(list(map(np.asarray, phis)))


def csg_diff(phi_a: Any, phi_b: Any) -> Any:
    """A \\ B: max(phi_a, -phi_b)."""
    return np.maximum(np.asarray(phi_a), -np.asarray(phi_b))


def composed(phi_funcs: list[Callable], op: str = "union") -> Callable:
    """Compose multiple SDF functions p → φ via `op`."""
    if op == "union":
        return lambda p: csg_union(*map(lambda f: f(p), phi_funcs))
    if op == "intersect":
        return lambda p: csg_intersect(*map(lambda f: f(p), phi_funcs))
    raise ValueError(f"unknown op: {op}")


__all__ = ["composed", "csg_diff", "csg_intersect", "csg_union"]
