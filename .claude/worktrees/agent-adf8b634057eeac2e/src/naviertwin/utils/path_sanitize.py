"""Path sanitization — prevent traversal (..) escapes.

Examples:
    >>> from pathlib import Path
    >>> from naviertwin.utils.path_sanitize import safe_join
    >>> safe_join('/tmp', 'foo/bar.txt').name
    'bar.txt'
"""

from __future__ import annotations

from pathlib import Path


def safe_join(root: str | Path, user_path: str) -> Path:
    """Resolve and assert resulting path stays inside `root`."""
    root_p = Path(root).resolve()
    candidate = (root_p / user_path).resolve()
    try:
        candidate.relative_to(root_p)
    except ValueError as e:
        raise ValueError(f"path escapes root: {user_path}") from e
    return candidate


__all__ = ["safe_join"]
