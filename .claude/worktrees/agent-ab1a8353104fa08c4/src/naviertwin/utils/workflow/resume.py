"""Resume-from-checkpoint hook — find latest matching ckpt.

Examples:
    >>> import tempfile, pathlib
    >>> from naviertwin.utils.workflow.resume import find_latest_ckpt
    >>> with tempfile.TemporaryDirectory() as d:
    ...     (pathlib.Path(d) / 'ckpt_0010.bin').write_bytes(b'')
    ...     (pathlib.Path(d) / 'ckpt_0020.bin').write_bytes(b'')
    ...     find_latest_ckpt(d).name
    'ckpt_0020.bin'
"""

from __future__ import annotations

from pathlib import Path


def find_latest_ckpt(
    root: str | Path, *, pattern: str = "ckpt_*.bin",
) -> Path | None:
    files = sorted(Path(root).glob(pattern))
    return files[-1] if files else None


__all__ = ["find_latest_ckpt"]
