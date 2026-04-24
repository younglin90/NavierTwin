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
