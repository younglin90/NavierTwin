"""Round 441 — path sanitize."""

from __future__ import annotations

import pytest


class TestPathSanitize:
    def test_normal(self, tmp_path) -> None:
        from naviertwin.utils.path_sanitize import safe_join

        p = safe_join(tmp_path, "foo/bar.txt")
        assert p.name == "bar.txt"
        assert str(p).startswith(str(tmp_path))

    def test_escape_blocked(self, tmp_path) -> None:
        from naviertwin.utils.path_sanitize import safe_join

        with pytest.raises(ValueError):
            safe_join(tmp_path, "../../etc/passwd")
