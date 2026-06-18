"""DAG runner — topological sort + execute callable per node.

Examples:
    >>> from naviertwin.utils.workflow.dag_runner import DAGRunner
    >>> r = DAGRunner()
    >>> r.add('a', lambda inputs: 1)
    >>> r.add('b', lambda inputs: inputs['a'] + 1, deps=['a'])
    >>> r.run()['b']
    2
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class DAGRunner:
    def __init__(self) -> None:
        self.nodes: dict[str, Callable[[dict], Any]] = {}
        self.deps: dict[str, list[str]] = {}

    def add(self, name: str, fn: Callable[[dict], Any], *,
            deps: list[str] | None = None) -> None:
        self.nodes[name] = fn
        self.deps[name] = list(deps or [])

    def topo(self) -> list[str]:
        order: list[str] = []
        visited: set[str] = set()

        def dfs(n: str) -> None:
            if n in visited:
                return
            deps = self.deps.get(n, [])
            dep_idx = 0
            while dep_idx < len(deps):
                d = deps[dep_idx]
                dfs(d)
                dep_idx += 1
            visited.add(n)
            order.append(n)

        node_names = list(self.nodes)
        node_idx = 0
        while node_idx < len(node_names):
            n = node_names[node_idx]
            dfs(n)
            node_idx += 1
        return order

    def run(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        order = self.topo()
        idx = 0
        while idx < len(order):
            n = order[idx]
            results[n] = self.nodes[n](results)
            idx += 1
        return results


__all__ = ["DAGRunner"]
