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
            lines = f.readlines()
            idx = 0
            while idx < len(lines):
                line = lines[idx]
                line = line.strip()
                if line:
                    out.append(json.loads(line))
                idx += 1
        return out

    def filter(self, **match: Any) -> list[dict[str, Any]]:
        records = self.read_all()
        items = list(match.items())
        out: list[dict[str, Any]] = []
        record_idx = 0
        while record_idx < len(records):
            r = records[record_idx]
            item_idx = 0
            matched = True
            while item_idx < len(items):
                k, v = items[item_idx]
                if r.get(k) != v:
                    matched = False
                    break
                item_idx += 1
            if matched:
                out.append(r)
            record_idx += 1
        return out


__all__ = ["StructuredLogger"]
