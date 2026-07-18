"""Round 352 — GitHub Actions CI gen."""

from __future__ import annotations


class TestCIGen:
    def test_yaml_contents(self) -> None:
        from naviertwin.utils.github_ci_gen import ci_yaml

        text = ci_yaml(versions=["3.11"])
        assert "name: CI" in text
        assert "pytest" in text
        assert "3.11" in text

    def test_write(self, tmp_path) -> None:
        from naviertwin.utils.github_ci_gen import write_ci

        p = tmp_path / ".github/workflows/ci.yml"
        write_ci(p)
        assert p.exists()
        assert "pytest" in p.read_text()
