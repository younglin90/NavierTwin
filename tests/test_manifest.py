"""Round 96 — 실행 매니페스트."""

from __future__ import annotations

import json
from pathlib import Path


class TestManifest:
    def test_build(self) -> None:
        from naviertwin.core.digital_twin.manifest import build_manifest

        m = build_manifest(
            reducer="pod", n_modes=5, surrogate="rbf",
            metrics={"rmse": 0.01},
            extra={"case": "pipe"},
        )
        assert m["config"]["n_modes"] == 5
        assert "timestamp" in m
        assert "environment" in m
        assert m["extra"]["case"] == "pipe"

    def test_save(self, tmp_path: Path) -> None:
        from naviertwin.core.digital_twin.manifest import (
            build_manifest,
            save_manifest,
        )

        m = build_manifest(reducer="pod", n_modes=3, surrogate="kriging")
        p = save_manifest(m, tmp_path / "manifest.json")
        assert p.exists()
        back = json.loads(p.read_text())
        assert back["config"]["surrogate"] == "kriging"
