"""Post-Process 실행 이력 — 최근 실행 기록 + 메타 캐시.

Facade 호출 이력을 메모리에 보관해 GUI에서 재실행 / 비교 가능.
디스크 직렬화 (JSON) 지원.

Examples:
    >>> from naviertwin.core.post_process_history import RunHistory
    >>> import numpy as np
    >>> hist = RunHistory(max_entries=10)
    >>> hist.record("psd_welch", {"fs": 100.0}, {"frequency": np.zeros(5)},
    ...             status="ok")
    >>> len(hist)
    1
"""

from __future__ import annotations

import datetime as _dt
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class RunHistory:
    """일정 길이의 실행 이력 deque.

    각 항목은 dict {timestamp, op, kwargs_summary, status, result_summary, error}.
    """

    def __init__(self, max_entries: int = 50) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries > 0, got {max_entries}")
        self._entries: deque = deque(maxlen=max_entries)

    def __len__(self) -> int:
        return len(self._entries)

    def record(
        self,
        op: str,
        kwargs: dict[str, Any],
        result: dict[str, Any] | None,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        """새 실행 항목 추가.

        Args:
            op: op 이름.
            kwargs: 호출 인자.
            result: 결과 dict (실패 시 None).
            status: "ok" | "error".
            error: 에러 메시지 (실패 시).
        """
        entry = {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "op": op,
            "kwargs_summary": _summarize_kwargs(kwargs),
            "status": status,
            "error": error,
            "result_summary": _summarize_result(result) if result else None,
        }
        self._entries.append(entry)

    def entries(self) -> list[dict[str, Any]]:
        """전체 이력 리스트 (최신이 마지막)."""
        return list(self._entries)

    def last(self) -> dict[str, Any] | None:
        return self._entries[-1] if self._entries else None

    def clear(self) -> None:
        self._entries.clear()

    def filter_by_op(self, op: str) -> list[dict[str, Any]]:
        return list(filter(lambda e: e["op"] == op, self._entries))

    def filter_by_status(self, status: str) -> list[dict[str, Any]]:
        return list(filter(lambda e: e["status"] == status, self._entries))

    def save_json(self, path: str | Path) -> Path:
        """이력을 JSON으로 저장."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.entries(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load_json(cls, path: str | Path, max_entries: int = 50) -> "RunHistory":
        """저장된 이력을 복원."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        hist = cls(max_entries=max(max_entries, len(data)))
        hist._entries.extend(data)
        return hist


def _summarize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """kwargs를 GUI/JSON에 표시하기 좋게 요약."""
    def summarize_item(item: tuple[str, Any]) -> tuple[str, Any]:
        k, v = item
        if isinstance(v, np.ndarray):
            return k, f"ndarray{tuple(v.shape)}"
        if isinstance(v, (list, tuple)):
            return k, f"{type(v).__name__}(len={len(v)})"
        if isinstance(v, (int, float, str, bool)):
            return k, v
        if callable(v):
            return k, f"<callable {getattr(v, '__name__', '?')}>"
        return k, type(v).__name__

    return dict(map(summarize_item, kwargs.items()))


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    """결과 dict를 요약 (큰 ndarray는 통계만)."""
    def summarize_item(item: tuple[str, Any]) -> tuple[str, Any]:
        k, v = item
        if isinstance(v, np.ndarray):
            if v.size > 0:
                return k, {
                    "shape": list(v.shape),
                    "min": float(v.min()),
                    "max": float(v.max()),
                    "mean": float(v.mean()),
                }
            return k, {"shape": [0]}
        if isinstance(v, (int, float, str, bool)):
            return k, v
        if isinstance(v, dict):
            return k, f"dict(keys={list(v.keys())[:5]})"
        if isinstance(v, list):
            return k, f"list(len={len(v)})"
        return k, type(v).__name__

    return dict(map(summarize_item, result.items()))


__all__ = ["RunHistory"]
