"""Round 70 — Sphinx docs 구조 검증 (빌드 없이)."""

from __future__ import annotations

import runpy
from argparse import _SubParsersAction
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"


class TestDocsStructure:
    def test_conf_exists(self) -> None:
        assert (DOCS / "source" / "conf.py").exists()

    def test_conf_release_tracks_package_version(self) -> None:
        """Sphinx release metadata must track the package version."""
        from naviertwin import __version__

        namespace = runpy.run_path(str(DOCS / "source" / "conf.py"))

        assert namespace["release"] == __version__
        assert namespace["version"] == ".".join(__version__.split(".")[:2])

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

    def test_docs_cli_reference_lists_public_subcommands(self) -> None:
        """CLI reference must stay in sync with parser public subcommands."""
        from naviertwin.main import _build_parser

        parser = _build_parser()
        subparsers = [
            action for action in parser._actions if isinstance(action, _SubParsersAction)
        ]
        assert len(subparsers) == 1

        cli = DOCS / "source" / "cli.rst"
        index = DOCS / "source" / "index.rst"
        content = cli.read_text(encoding="utf-8")
        index_content = index.read_text(encoding="utf-8")

        assert "   cli" in index_content
        assert "naviertwin --version" in content
        for command in sorted(subparsers[0].choices):
            assert command in content
            assert f"naviertwin {command}" in content
        assert "Expected:" in content

    def test_customer_docs_do_not_hardcode_current_package_version(self) -> None:
        """Customer-facing docs should not stale when the package version changes."""
        from naviertwin import __version__

        customer_docs = [
            ROOT / "README.md",
            DOCS / "source" / "cli.rst",
        ]

        for path in customer_docs:
            assert __version__ not in path.read_text(encoding="utf-8")

    def test_gui_docs_track_customer_facing_tabs_and_tools(self) -> None:
        """GUI docs must reflect currently exposed commercial desktop workflows."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        gui = (DOCS / "source" / "gui.rst").read_text(encoding="utf-8")
        overview = (DOCS / "source" / "overview.rst").read_text(encoding="utf-8")
        combined = readme + "\n" + gui + "\n" + overview

        assert "10 탭" in readme
        assert "10 개 탭 구조" in gui
        assert "PySide6 기반 10 탭 인터페이스" in overview
        for token in [
            "Post-Tools",
            "Explain",
            "Kernel SHAP",
            "Attention viz",
            "Active Learning",
            "SINDy",
            "ONNX",
            "TorchScript",
            "파이프라인 데모",
            "API 서버",
        ]:
            assert token in combined
        assert "GUI: Import / Analyze / Reduce / Model / Twin / Export 6 탭" not in readme
        assert "8 개 탭 구조" not in gui
        assert "6+ 탭 인터페이스" not in overview
