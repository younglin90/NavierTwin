"""Round 69 — config loader + overrides."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestDotted:
    def test_set_get(self) -> None:
        from naviertwin.utils.config_loader import get_dotted, set_dotted

        cfg: dict = {}
        set_dotted(cfg, "a.b.c", 42)
        assert cfg["a"]["b"]["c"] == 42
        assert get_dotted(cfg, "a.b.c") == 42
        assert get_dotted(cfg, "missing", default="X") == "X"


class TestOverrides:
    def test_scalar_types(self) -> None:
        from naviertwin.utils.config_loader import merge_overrides

        base = {"lr": 1e-3, "seed": 0, "model": {"hidden": 32}}
        out = merge_overrides(base, [
            "lr=0.0005",
            "seed=42",
            "model.hidden=128",
            "use_gpu=true",
            "name=demo",
        ])
        assert out["lr"] == 0.0005
        assert out["seed"] == 42
        assert out["model"]["hidden"] == 128
        assert out["use_gpu"] is True
        assert out["name"] == "demo"

    def test_invalid_override_ignored(self) -> None:
        from naviertwin.utils.config_loader import merge_overrides

        out = merge_overrides({"x": 1}, ["no_equals_here"])
        assert out == {"x": 1}


class TestLoadSave:
    def test_json_roundtrip(self, tmp_path: Path) -> None:
        from naviertwin.utils.config_loader import load_config, save_config

        cfg = {"lr": 1e-3, "model": {"hidden": 32, "dropout": 0.1}}
        f = save_config(cfg, tmp_path / "cfg.json")
        cfg2 = load_config(f)
        assert cfg2 == cfg

    def test_yaml_roundtrip(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        from naviertwin.utils.config_loader import load_config, save_config

        cfg = {"lr": 1e-3, "model": {"hidden": 32}}
        f = save_config(cfg, tmp_path / "cfg.yaml")
        cfg2 = load_config(f)
        assert cfg2 == cfg

    def test_toml_write(self, tmp_path: Path) -> None:
        from naviertwin.utils.config_loader import load_config, save_config

        cfg = {"lr": 1e-3, "name": "exp1", "model": {"hidden": 32}}
        f = save_config(cfg, tmp_path / "cfg.toml")
        # Python 3.11+ tomllib
        import sys
        if sys.version_info >= (3, 11):
            cfg2 = load_config(f)
            assert cfg2["lr"] == 1e-3
            assert cfg2["model"]["hidden"] == 32

    def test_load_with_overrides(self, tmp_path: Path) -> None:
        from naviertwin.utils.config_loader import load_config, save_config

        cfg = {"lr": 1e-3, "model": {"hidden": 32}}
        f = save_config(cfg, tmp_path / "cfg.json")
        merged = load_config(f, overrides=["lr=1e-5", "model.hidden=64"])
        assert merged["lr"] == 1e-5
        assert merged["model"]["hidden"] == 64

    def test_unknown_format(self, tmp_path: Path) -> None:
        from naviertwin.utils.config_loader import save_config

        with pytest.raises(ValueError):
            save_config({"x": 1}, tmp_path / "cfg.xyz")

    def test_missing_file(self) -> None:
        from naviertwin.utils.config_loader import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_empty_path_returns_empty(self) -> None:
        from naviertwin.utils.config_loader import load_config

        assert load_config(None, ["x=1"])["x"] == 1
