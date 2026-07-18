"""HTML 리포트 — matplotlib 차트를 base64 inline embed.

Examples:
    >>> from naviertwin.core.report.html_report import HTMLReport
    >>> r = HTMLReport("Test")
    >>> r.para("hello")  # doctest: +ELLIPSIS
    <naviertwin.core.report.html_report.HTMLReport object at ...>
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any


class HTMLReport:
    def __init__(self, title: str = "Report") -> None:
        self.title = title
        self._body: list[str] = []

    def h1(self, t: str) -> "HTMLReport":
        self._body.append(f"<h1>{t}</h1>")
        return self

    def h2(self, t: str) -> "HTMLReport":
        self._body.append(f"<h2>{t}</h2>")
        return self

    def para(self, t: str) -> "HTMLReport":
        self._body.append(f"<p>{t}</p>")
        return self

    def table(self, headers: list[str], rows: list[list[Any]]) -> "HTMLReport":
        thead = "".join(map(lambda h: f"<th>{h}</th>", headers))
        tbody = "".join(
            map(
                lambda row: "<tr>"
                + "".join(map(lambda c: f"<td>{c}</td>", row))
                + "</tr>",
                rows,
            )
        )
        self._body.append(
            f'<table border="1" cellpadding="4">'
            f"<thead><tr>{thead}</tr></thead>"
            f"<tbody>{tbody}</tbody></table>"
        )
        return self

    def figure(self, fig: Any, caption: str = "") -> "HTMLReport":
        """matplotlib Figure → base64 PNG 인라인."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        self._body.append(
            f'<figure><img src="data:image/png;base64,{b64}" />{cap}</figure>'
        )
        return self

    def render(self) -> str:
        style = (
            "<style>body{font-family:sans-serif;margin:2em;}"
            "table{border-collapse:collapse;}"
            "img{max-width:100%;}</style>"
        )
        body = "\n".join(self._body)
        return (
            f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{self.title}</title>{style}</head>"
            f"<body>{body}</body></html>"
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render(), encoding="utf-8")
        return p


__all__ = ["HTMLReport"]
