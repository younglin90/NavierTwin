"""파일 tail 리더 — 실시간 CFD 로그/잔차 스트리밍.

Examples:
    >>> # TailReader 로 파일에 append 되는 줄을 subscriber 에게 전달
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable


class TailReader:
    """파일을 주기적으로 읽어 새 라인을 handler 에 전달."""

    def __init__(
        self, path: str | Path,
        handler: Callable[[str], None],
        *, poll_interval: float = 0.1,
    ) -> None:
        self.path = Path(path)
        self.handler = handler
        self.poll_interval = float(poll_interval)
        self._running = False
        self._thread: threading.Thread | None = None
        self._pos = 0

    def _loop(self) -> None:
        while self._running:
            try:
                if self.path.exists():
                    with self.path.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(self._pos)
                        while True:
                            line = f.readline()
                            if not line:
                                break
                            if line.endswith("\n"):
                                self.handler(line.rstrip("\n"))
                                self._pos = f.tell()
                            else:
                                break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.poll_interval)

    def start(self) -> "TailReader":
        self._running = True
        # 기존 내용 건너뜀
        if self.path.exists():
            self._pos = self.path.stat().st_size
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def __enter__(self) -> "TailReader":
        return self.start()

    def __exit__(self, *exc) -> None:  # noqa: ANN001
        self.stop()


__all__ = ["TailReader"]
