"""원자적 파일 쓰기 — tempfile + os.replace 로 부분 파일 노출 방지.

Examples:
    >>> from pathlib import Path
    >>> from naviertwin.utils.atomic_io import atomic_write_text
    >>> p = Path("/tmp/x.txt")
    >>> atomic_write_text(p, "hello")
    >>> p.read_text()
    'hello'
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def atomic_write_text(
    path: str | Path, text: str, *, encoding: str = "utf-8"
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


@contextmanager
def atomic_open(
    path: str | Path, mode: str = "w", *, encoding: str | None = "utf-8",
) -> Iterator:
    """with 블록 안에서 실패 시 최종 파일은 갱신되지 않음."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp",
    )
    kwargs = {} if "b" in mode else {"encoding": encoding}
    try:
        f = os.fdopen(fd, mode, **kwargs)  # type: ignore[arg-type]
        try:
            yield f
        finally:
            f.flush()
            os.fsync(f.fileno())
            f.close()
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


__all__ = ["atomic_write_text", "atomic_write_bytes", "atomic_open"]
