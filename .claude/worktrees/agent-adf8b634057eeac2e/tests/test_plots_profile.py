"""Round 66 — publication plots + profile 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestPlots:
    def test_apply_style(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib as mpl

        from naviertwin.core.report.plots import apply_publication_style

        apply_publication_style(font_size=14)
        assert mpl.rcParams["font.size"] == 14
        assert mpl.rcParams["axes.grid"] is True

    def test_loss_curve(self) -> None:
        pytest.importorskip("matplotlib")
        from naviertwin.core.report.plots import plot_loss_curve

        fig = plot_loss_curve({"FNO": [1.0, 0.5, 0.1], "POD": [0.8, 0.3, 0.05]})
        assert fig is not None
        assert len(fig.axes[0].lines) == 2

    def test_compare_metrics(self) -> None:
        pytest.importorskip("matplotlib")
        from naviertwin.core.report.plots import plot_compare_metrics

        results = {
            "Kriging": {"rmse": 0.02, "r2": 0.98},
            "RBF": {"rmse": 0.03, "r2": 0.96},
        }
        fig = plot_compare_metrics(results, metric="rmse")
        assert fig is not None

    def test_field_2d(self) -> None:
        pytest.importorskip("matplotlib")
        from naviertwin.core.report.plots import plot_field_2d

        rng = np.random.default_rng(0)
        field = rng.standard_normal((16, 16))
        fig = plot_field_2d(field, title="Test")
        assert fig is not None

    def test_pod_energy(self) -> None:
        pytest.importorskip("matplotlib")
        from naviertwin.core.report.plots import plot_pod_energy

        s = np.array([10.0, 5.0, 2.0, 0.5, 0.1])
        fig = plot_pod_energy(s)
        assert fig is not None


class TestProfile:
    def test_set_get(self) -> None:
        from naviertwin.utils.profile import Profile

        p = Profile()
        p.set("seed", 42)
        p.set("custom", "value")
        assert p.get("seed") == 42
        assert p.get("custom") == "value"
        assert p.get("missing", "default") == "default"

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        from naviertwin.utils.profile import Profile

        p = Profile(seed=7, lr=5e-4)
        p.set("batch_size", 32)
        f = p.save(tmp_path / "profile.json")
        assert f.exists()

        p2 = Profile.load(f)
        assert p2.seed == 7
        assert p2.lr == 5e-4
        assert p2.get("batch_size") == 32

    def test_apply_seed_stable(self) -> None:
        from naviertwin.utils.profile import Profile

        p = Profile(seed=123)
        p.apply_seed()
        a = np.random.random(3)
        p.apply_seed()
        b = np.random.random(3)
        assert np.allclose(a, b)

    def test_meta_has_deps(self) -> None:
        from naviertwin.utils.profile import Profile

        p = Profile()
        meta = p.to_dict()["_meta"]
        assert "python" in meta
        assert "dependencies" in meta
        assert "numpy" in meta["dependencies"]
