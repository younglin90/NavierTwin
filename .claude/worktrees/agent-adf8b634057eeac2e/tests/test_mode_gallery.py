"""Round 242 — mode gallery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestGallery:
    def test_render(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        from naviertwin.gui.mode_gallery import render_mode_gallery

        rng = np.random.default_rng(0)
        modes = rng.standard_normal((16 * 16, 6))
        p = render_mode_gallery(modes, (16, 16), tmp_path / "gal.png", cols=3)
        assert p.exists()
        assert p.stat().st_size > 500
