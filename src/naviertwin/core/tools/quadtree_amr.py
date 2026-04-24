"""2D quadtree adaptive mesh refinement — 지표 기반 셀 분할.

Examples:
    >>> from naviertwin.core.tools.quadtree_amr import QuadCell, refine_tree
    >>> root = QuadCell(0, 0, 1, 1)
    >>> root.split()
    >>> len(root.children)
    4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class QuadCell:
    x0: float
    y0: float
    x1: float
    y1: float
    level: int = 0
    children: list["QuadCell"] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5)

    @property
    def size(self) -> tuple[float, float]:
        return (self.x1 - self.x0, self.y1 - self.y0)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def split(self) -> None:
        mx = 0.5 * (self.x0 + self.x1)
        my = 0.5 * (self.y0 + self.y1)
        L = self.level + 1
        self.children = [
            QuadCell(self.x0, self.y0, mx, my, L),
            QuadCell(mx, self.y0, self.x1, my, L),
            QuadCell(self.x0, my, mx, self.y1, L),
            QuadCell(mx, my, self.x1, self.y1, L),
        ]

    def leaves(self) -> list["QuadCell"]:
        if self.is_leaf:
            return [self]
        out: list[QuadCell] = []
        for c in self.children:
            out.extend(c.leaves())
        return out


def refine_tree(
    root: QuadCell,
    indicator: Callable[[QuadCell], float],
    *, threshold: float = 0.1, max_level: int = 5,
) -> None:
    """indicator(cell) > threshold 이면 분할 (최대 max_level)."""
    stack = [root]
    while stack:
        cell = stack.pop()
        if cell.level >= max_level:
            continue
        if cell.is_leaf and float(indicator(cell)) > threshold:
            cell.split()
            stack.extend(cell.children)


def leaf_count(root: QuadCell) -> int:
    return len(root.leaves())


__all__ = ["QuadCell", "refine_tree", "leaf_count"]
