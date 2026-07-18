"""Round 351 — Dockerfile generator."""

from __future__ import annotations


class TestDocker:
    def test_text_has_from(self) -> None:
        from naviertwin.utils.docker_gen import dockerfile_text

        text = dockerfile_text(cuda_version="12.1.0")
        assert text.startswith("FROM")
        assert "12.1.0" in text
        assert "naviertwin" in text

    def test_extras(self) -> None:
        from naviertwin.utils.docker_gen import dockerfile_text

        text = dockerfile_text(extras="full")
        assert "[full]" in text

    def test_write(self, tmp_path) -> None:
        from naviertwin.utils.docker_gen import write_dockerfile

        p = tmp_path / "Dockerfile"
        write_dockerfile(p, cuda_version="11.8.0")
        assert "11.8.0" in p.read_text()
