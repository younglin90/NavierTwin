"""구조화 JSON 로그 — JSONL 파일에 event 저장.

Examples:
    >>> from pathlib import Path
    >>> from naviertwin.utils.structured_log import StructuredLogger
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredLogger:
    """append-only JSONL event logger (thread-safe)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(
        self, event: str, *, level: str = "info", **fields: Any,
    ) -> None:
        rec = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            **fields,
        }
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def filter(self, **match: Any) -> list[dict[str, Any]]:
        return [
            r for r in self.read_all()
            if all(r.get(k) == v for k, v in match.items())
        ]


__all__ = ["StructuredLogger"]
