"""Round 70 — Sphinx docs 구조 검증 (빌드 없이)."""

from __future__ import annotations

from pathlib import Path


DOCS = Path(__file__).parent.parent / "docs"


class TestDocsStructure:
    def test_conf_exists(self) -> None:
        assert (DOCS / "source" / "conf.py").exists()

    def test_index_exists(self) -> None:
        idx = DOCS / "source" / "index.rst"
        assert idx.exists()
        content = idx.read_text(encoding="utf-8")
        assert "toctree" in content

    def test_api_stubs(self) -> None:
        api = DOCS / "source" / "api"
        assert api.exists()
        # 주요 패키지 존재
        for pkg in ["cfd_reader", "dimensionality_reduction", "operator_learning"]:
            assert (api / f"{pkg}.rst").exists()

    def test_makefile(self) -> None:
        mf = DOCS / "Makefile"
        assert mf.exists()
        content = mf.read_text(encoding="utf-8")
        assert "html:" in content

    def test_overview_has_korean(self) -> None:
        ov = DOCS / "source" / "overview.rst"
        content = ov.read_text(encoding="utf-8").replace("\n", " ")
        assert "디지털" in content and "트윈" in content
