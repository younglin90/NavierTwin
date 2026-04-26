"""Round 589 — utils.config validation + load/save round-trip coverage."""

from __future__ import annotations

import pytest


class TestNavierTwinConfig:
    def test_default_init(self) -> None:
        from naviertwin.utils.config import NavierTwinConfig

        c = NavierTwinConfig()
        assert c.log_level == "INFO"
        assert c.language == "ko"
        assert c.theme == "dark"
        assert c.gpu_enabled is True

    def test_invalid_log_level(self) -> None:
        from naviertwin.utils.config import NavierTwinConfig

        with pytest.raises(ValueError, match="log_level"):
            NavierTwinConfig(log_level="VERBOSE")

    def test_invalid_language(self) -> None:
        from naviertwin.utils.config import NavierTwinConfig

        with pytest.raises(ValueError, match="language"):
            NavierTwinConfig(language="ja")

    def test_invalid_theme(self) -> None:
        from naviertwin.utils.config import NavierTwinConfig

        with pytest.raises(ValueError, match="theme"):
            NavierTwinConfig(theme="solar")

    def test_negative_threads(self) -> None:
        from naviertwin.utils.config import NavierTwinConfig

        with pytest.raises(ValueError, match="max_threads"):
            NavierTwinConfig(max_threads=-1)

    def test_recent_projects_truncated(self) -> None:
        from naviertwin.utils.config import NavierTwinConfig

        c = NavierTwinConfig(recent_projects=[f"/p/{i}" for i in range(15)])
        assert len(c.recent_projects) == 10
        assert c.recent_projects[0] == "/p/5"

    def test_save_load_round_trip(self, tmp_path) -> None:
        from naviertwin.utils.config import (
            NavierTwinConfig,
            load_config,
            save_config,
        )

        c = NavierTwinConfig(log_level="DEBUG", theme="light", max_threads=4)
        path = tmp_path / "cfg.json"
        save_config(c, path)
        c2 = load_config(path)
        assert c2.log_level == "DEBUG"
        assert c2.theme == "light"
        assert c2.max_threads == 4

    def test_load_missing_returns_default(self, tmp_path) -> None:
        from naviertwin.utils.config import load_config

        c = load_config(tmp_path / "missing.json")
        # default values
        assert c.log_level == "INFO"

    def test_load_ignores_unknown_keys(self, tmp_path) -> None:
        from naviertwin.utils.config import load_config

        path = tmp_path / "cfg.json"
        path.write_text('{"log_level": "WARNING", "bogus_key": 42}')
        c = load_config(path)
        assert c.log_level == "WARNING"
