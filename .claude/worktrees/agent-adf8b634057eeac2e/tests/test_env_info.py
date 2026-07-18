"""Round 92 — 런타임 환경 정보."""

from __future__ import annotations


class TestEnvInfo:
    def test_basics(self) -> None:
        from naviertwin.utils.env_info import collect_env

        info = collect_env()
        assert "python" in info
        assert "platform" in info
        assert "numpy" in info
        assert "cuda_available" in info

    def test_format(self) -> None:
        from naviertwin.utils.env_info import collect_env, format_env

        s = format_env(collect_env())
        assert "python" in s
        assert ":" in s

    def test_collect_env_versions_do_not_import_packages(self, monkeypatch) -> None:
        import importlib

        import naviertwin.utils.env_info as env_info

        def fail_import_module(name: str):
            raise AssertionError(f"unexpected package import: {name}")

        monkeypatch.setattr(importlib, "import_module", fail_import_module)
        monkeypatch.setattr(env_info, "_DISTRIBUTION_NAMES", {"fakepkg": "fake-dist"})
        monkeypatch.setattr(env_info.metadata, "version", lambda name: "1.2.3")

        assert env_info._pkg_version("fakepkg") == "1.2.3"
