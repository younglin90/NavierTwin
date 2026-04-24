"""Round 248 — safe TOML."""

from __future__ import annotations

from pathlib import Path


class TestTOML:
    def test_load(self) -> None:
        from naviertwin.utils.safe_toml import toml_loads

        d = toml_loads('x = 1\ny = "hi"\n[sec]\nz = 2.5\n')
        assert d["x"] == 1
        assert d["y"] == "hi"
        assert d["sec"]["z"] == 2.5

    def test_dump_roundtrip(self, tmp_path: Path) -> None:
        from naviertwin.utils.safe_toml import (
            toml_dump,
            toml_load_file,
        )

        p = tmp_path / "c.toml"
        data = {"x": 1, "b": True, "s": "hello", "nested": {"y": 2.5}}
        p.write_text(toml_dump(data))
        back = toml_load_file(p)
        assert back["x"] == 1
        assert back["b"] is True
        assert back["s"] == "hello"
        assert back["nested"]["y"] == 2.5
