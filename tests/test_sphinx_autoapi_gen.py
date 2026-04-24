"""Round 354 — Sphinx autoapi gen."""

from __future__ import annotations


class TestSphinx:
    def test_conf_has_autoapi(self) -> None:
        from naviertwin.utils.sphinx_autoapi_gen import sphinx_conf

        text = sphinx_conf(project="MyProj", src_path="../src")
        assert "autoapi" in text
        assert "MyProj" in text
        assert "../src" in text

    def test_write(self, tmp_path) -> None:
        from naviertwin.utils.sphinx_autoapi_gen import write_sphinx_conf

        p = tmp_path / "docs/conf.py"
        write_sphinx_conf(p)
        assert p.exists()
        assert "autoapi" in p.read_text()
