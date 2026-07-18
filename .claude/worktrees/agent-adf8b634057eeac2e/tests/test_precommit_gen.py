"""Round 353 — pre-commit config gen."""

from __future__ import annotations


class TestPrecommit:
    def test_yaml_has_ruff_mypy(self) -> None:
        from naviertwin.utils.precommit_gen import precommit_yaml

        text = precommit_yaml(ruff="0.7.0", mypy="1.11.0")
        assert "ruff" in text
        assert "mypy" in text
        assert "0.7.0" in text
        assert "1.11.0" in text

    def test_write(self, tmp_path) -> None:
        from naviertwin.utils.precommit_gen import write_precommit

        p = tmp_path / ".pre-commit-config.yaml"
        write_precommit(p)
        assert p.exists()
        assert "ruff" in p.read_text()
