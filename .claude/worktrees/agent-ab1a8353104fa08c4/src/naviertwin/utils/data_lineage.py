"""Data lineage — DAG of artifact derivations.

Examples:
    >>> from naviertwin.utils.data_lineage import LineageDAG
    >>> g = LineageDAG()
    >>> g.add('raw'); g.add('clean', parents=['raw'])
    >>> g.ancestors('clean')
    ['raw']
"""

from __future__ import annotations


class LineageDAG:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, list[str]] = {}  # child -> parents

    def add(self, name: str, *, parents: list[str] | None = None,
            meta: dict | None = None) -> None:
        if name in self.nodes:
            raise KeyError(name)
        self.nodes[name] = meta or {}
        self.edges[name] = list(parents or [])

    def ancestors(self, name: str) -> list[str]:
        seen: list[str] = []
        stack = list(self.edges.get(name, []))
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.append(n)
            stack.extend(self.edges.get(n, []))
        return seen


__all__ = ["LineageDAG"]
