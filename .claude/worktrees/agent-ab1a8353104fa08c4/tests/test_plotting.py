"""Round 152 — 플롯 헬퍼."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestPlot:
    def test_field_2d(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        from naviertwin.gui.plotting import plot_field_2d

        p = plot_field_2d(
            np.random.default_rng(0).standard_normal((20, 20)),
            tmp_path / "f.png", title="t", cmap="viridis",
        )
        assert p.exists()
        assert p.stat().st_size > 100

    def test_line(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        from naviertwin.gui.plotting import plot_line

        x = np.linspace(0, 1, 50)
        p = plot_line(x, np.sin(x * 6), tmp_path / "l.png", title="sin")
        assert p.exists()

    def test_loss(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        from naviertwin.gui.plotting import plot_loss_curve

        p = plot_loss_curve([1.0, 0.8, 0.5, 0.1, 0.01], tmp_path / "loss.png")
        assert p.exists()
