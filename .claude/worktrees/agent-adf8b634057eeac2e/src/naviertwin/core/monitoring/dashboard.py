"""대시보드 데이터 집계기 — 실시간 메트릭 ring buffer + 요약.

Examples:
    >>> from naviertwin.core.monitoring.dashboard import DashboardAggregator
    >>> d = DashboardAggregator(buffer_size=10)
    >>> d.push("rmse", 0.1)
    >>> d.push("rmse", 0.05)
    >>> d.summary("rmse")["latest"]
    0.05
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class DashboardAggregator:
    """여러 메트릭을 deque 로 유지."""

    def __init__(self, buffer_size: int = 1000) -> None:
        self.buffer_size = int(buffer_size)
        self._data: dict[str, deque] = {}

    def push(self, metric: str, value: float) -> None:
        if metric not in self._data:
            self._data[metric] = deque(maxlen=self.buffer_size)
        self._data[metric].append(float(value))

    def push_many(self, values: dict[str, float]) -> None:
        def _push(item: tuple[str, float]) -> None:
            self.push(item[0], item[1])

        tuple(map(_push, values.items()))

    def summary(self, metric: str) -> dict[str, float]:
        if metric not in self._data or len(self._data[metric]) == 0:
            return {"count": 0}
        arr = np.asarray(self._data[metric])
        return {
            "count": int(arr.size),
            "latest": float(arr[-1]),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    def all_metrics(self) -> list[str]:
        return list(self._data.keys())

    def snapshot(self) -> dict[str, Any]:
        return dict(map(lambda metric: (metric, self.summary(metric)), self._data))

    def reset(self, metric: str | None = None) -> None:
        if metric is None:
            self._data.clear()
        else:
            self._data.pop(metric, None)


__all__ = ["DashboardAggregator"]
