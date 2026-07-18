"""Markdown 리포트 빌더 — 섹션, 테이블, 이미지 embed.

Examples:
    >>> from naviertwin.core.report.markdown import MarkdownReport
    >>> r = MarkdownReport("Test")
    >>> r.h1("Title").para("body").table(["a","b"], [[1,2],[3,4]])
    <naviertwin.core.report.markdown.MarkdownReport object at ...>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MarkdownReport:
    def __init__(self, title: str = "") -> None:
        self.lines: list[str] = []
        if title:
            self.lines.append(f"# {title}\n")

    def h1(self, text: str) -> "MarkdownReport":
        self.lines.append(f"\n# {text}\n")
        return self

    def h2(self, text: str) -> "MarkdownReport":
        self.lines.append(f"\n## {text}\n")
        return self

    def h3(self, text: str) -> "MarkdownReport":
        self.lines.append(f"\n### {text}\n")
        return self

    def para(self, text: str) -> "MarkdownReport":
        self.lines.append(f"{text}\n")
        return self

    def code(self, text: str, lang: str = "") -> "MarkdownReport":
        self.lines.append(f"```{lang}\n{text}\n```\n")
        return self

    def table(
        self, headers: list[str], rows: list[list[Any]],
    ) -> "MarkdownReport":
        self.lines.append("| " + " | ".join(map(str, headers)) + " |")
        self.lines.append("| " + " | ".join(map(lambda _: "---", headers)) + " |")
        self.lines.extend(
            map(lambda row: "| " + " | ".join(map(str, row)) + " |", rows)
        )
        self.lines.append("")
        return self

    def image(self, path: str, alt: str = "") -> "MarkdownReport":
        self.lines.append(f"![{alt}]({path})\n")
        return self

    def bullet(self, items: list[str]) -> "MarkdownReport":
        self.lines.extend(map(lambda it: f"- {it}", items))
        self.lines.append("")
        return self

    def hr(self) -> "MarkdownReport":
        self.lines.append("\n---\n")
        return self

    def render(self) -> str:
        return "\n".join(self.lines)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render(), encoding="utf-8")
        return p


__all__ = ["MarkdownReport"]
