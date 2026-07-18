"""Round 139 — HTML 리포트."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestHTML:
    def test_basic(self) -> None:
        from naviertwin.core.report.html_report import HTMLReport

        r = HTMLReport("T").h1("Head").para("body")
        html = r.render()
        assert "<h1>Head</h1>" in html
        assert "<p>body</p>" in html
        assert "<!doctype html>" in html

    def test_table(self) -> None:
        from naviertwin.core.report.html_report import HTMLReport

        r = HTMLReport().table(["a", "b"], [[1, 2], [3, 4]])
        html = r.render()
        assert "<th>a</th>" in html
        assert "<td>3</td>" in html

    def test_figure(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from naviertwin.core.report.html_report import HTMLReport

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        r = HTMLReport("FigTest").figure(fig, caption="plot")
        plt.close(fig)
        p = r.save(tmp_path / "r.html")
        assert p.exists()
        text = p.read_text()
        assert "data:image/png;base64," in text
        assert "<figcaption>plot</figcaption>" in text
