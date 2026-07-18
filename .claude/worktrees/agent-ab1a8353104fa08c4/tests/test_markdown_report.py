"""Round 138 — Markdown report."""

from __future__ import annotations

from pathlib import Path


class TestMarkdown:
    def test_build(self) -> None:
        from naviertwin.core.report.markdown import MarkdownReport

        r = (
            MarkdownReport("Title")
            .h2("Section")
            .para("body text")
            .bullet(["a", "b"])
            .table(["col1", "col2"], [[1, 2], [3, 4]])
            .code("print('hi')", lang="python")
            .hr()
            .image("img.png", "alt")
        )
        out = r.render()
        assert "# Title" in out
        assert "| col1 | col2 |" in out
        assert "print('hi')" in out
        assert "![alt](img.png)" in out

    def test_save(self, tmp_path: Path) -> None:
        from naviertwin.core.report.markdown import MarkdownReport

        p = MarkdownReport("X").para("hello").save(tmp_path / "report.md")
        assert p.exists()
        assert "# X" in p.read_text()
