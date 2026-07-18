"""Round 247 + 591 — safe YAML full coverage (fallback paths)."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest


class TestYAML:
    def test_load_basic(self) -> None:
        from naviertwin.utils.safe_yaml import safe_yaml_load

        d = safe_yaml_load("a: 1\nb: 2.5\nc: hello\n")
        assert d["a"] == 1
        assert abs(d["b"] - 2.5) < 1e-12
        assert d["c"] == "hello"

    def test_file(self, tmp_path: Path) -> None:
        from naviertwin.utils.safe_yaml import (
            safe_yaml_dump,
            safe_yaml_load_file,
        )

        p = tmp_path / "c.yaml"
        p.write_text(safe_yaml_dump({"x": 1, "y": True}))
        d = safe_yaml_load_file(p)
        assert d["x"] == 1
        assert d["y"] is True

    def test_null_and_bool(self) -> None:
        from naviertwin.utils.safe_yaml import safe_yaml_load

        d = safe_yaml_load("a: null\nb: true\nc: false\n")
        assert d["a"] is None
        assert d["b"] is True
        assert d["c"] is False


class TestParseScalar:
    def _ps(self, s: str):
        from naviertwin.utils.safe_yaml import _parse_scalar

        return _parse_scalar(s)

    def test_null_variants(self) -> None:
        assert self._ps("") is None
        assert self._ps("null") is None
        assert self._ps("~") is None
        assert self._ps("None") is None

    def test_bool_true_variants(self) -> None:
        assert self._ps("yes") is True
        assert self._ps("on") is True
        assert self._ps("True") is True

    def test_bool_false_variants(self) -> None:
        assert self._ps("no") is False
        assert self._ps("off") is False
        assert self._ps("False") is False

    def test_float(self) -> None:
        assert self._ps("3.14") == pytest.approx(3.14)
        assert self._ps("1e3") == pytest.approx(1000.0)

    def test_int(self) -> None:
        assert self._ps("42") == 42

    def test_quoted_string(self) -> None:
        assert self._ps('"hello"') == "hello"
        assert self._ps("'world'") == "world"
        assert self._ps("plain") == "plain"


class TestFallbackLoad:
    def _fl(self, text: str):
        from naviertwin.utils.safe_yaml import _fallback_load

        return _fallback_load(text)

    def test_basic_kv(self) -> None:
        assert self._fl("a: 1\nb: 2\n") == {"a": 1, "b": 2}

    def test_skips_comments_and_blanks(self) -> None:
        result = self._fl("# comment\n\na: hello\n")
        assert result == {"a": "hello"}

    def test_skips_lines_without_colon(self) -> None:
        result = self._fl("nocolon\na: 1\n")
        assert result == {"a": 1}

    def test_bool_and_null(self) -> None:
        result = self._fl("flag: true\nkey: null\n")
        assert result == {"flag": True, "key": None}


class TestNoYamlFallback:
    def _block_yaml(self, monkeypatch):
        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "yaml":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)

    def test_load_fallback(self, monkeypatch) -> None:
        from naviertwin.utils import safe_yaml

        self._block_yaml(monkeypatch)
        result = safe_yaml.safe_yaml_load("x: 10\ny: true\n")
        assert result["x"] == 10
        assert result["y"] is True

    def test_dump_dict_fallback(self, monkeypatch) -> None:
        from naviertwin.utils import safe_yaml

        self._block_yaml(monkeypatch)
        text = safe_yaml.safe_yaml_dump({"key": "value", "num": 42})
        assert "key: value" in text
        assert "num: 42" in text

    def test_dump_non_dict_fallback(self, monkeypatch) -> None:
        from naviertwin.utils import safe_yaml

        self._block_yaml(monkeypatch)
        text = safe_yaml.safe_yaml_dump([1, 2, 3])
        assert "[1, 2, 3]" in text
