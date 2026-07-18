"""Forest of octrees — Morton (Z-order) index encoding (3D).

Examples:
    >>> from naviertwin.core.amr.octree_forest import morton3, demorton3
    >>> idx = morton3(1, 2, 3)
    >>> demorton3(idx)
    (1, 2, 3)
"""

from __future__ import annotations


def _spread3(v: int) -> int:
    """Insert two 0-bits between each pair (3D Morton)."""
    v &= 0x1FFFFF  # 21 bits per axis
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


def _compact3(v: int) -> int:
    v &= 0x1249249249249249
    v = (v ^ (v >> 2)) & 0x10C30C30C30C30C3
    v = (v ^ (v >> 4)) & 0x100F00F00F00F00F
    v = (v ^ (v >> 8)) & 0x1F0000FF0000FF
    v = (v ^ (v >> 16)) & 0x1F00000000FFFF
    v = (v ^ (v >> 32)) & 0x1FFFFF
    return v


def morton3(x: int, y: int, z: int) -> int:
    """Z-order curve index from (x, y, z)."""
    return _spread3(x) | (_spread3(y) << 1) | (_spread3(z) << 2)


def demorton3(idx: int) -> tuple[int, int, int]:
    return _compact3(idx), _compact3(idx >> 1), _compact3(idx >> 2)


__all__ = ["demorton3", "morton3"]
