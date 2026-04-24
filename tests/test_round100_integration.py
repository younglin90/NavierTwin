"""Round 100 — 마일스톤: R76-99 신규 유틸 통합 임포트/기초 동작 스모크."""

from __future__ import annotations

import pytest

MODULES = [
    "naviertwin.core.digital_twin.batch_predict",
    "naviertwin.utils.training_callbacks",
    "naviertwin.core.sampling.param_sweep",
    "naviertwin.core.analysis.latent_embedding",
    "naviertwin.utils.rolling_stats",
    "naviertwin.utils.json_safe",
    "naviertwin.core.validation.field_diff",
    "naviertwin.core.cfd_reader.csv_snapshots",
    "naviertwin.core.augmentation.noise",
    "naviertwin.utils.retry",
    "naviertwin.core.preprocessing.sliding_window",
    "naviertwin.core.preprocessing.normalizer",
    "naviertwin.core.analysis.spectrum",
    "naviertwin.core.preprocessing.splitter",
    "naviertwin.core.analysis.dimensionless",
    "naviertwin.core.analysis.correlation",
    "naviertwin.utils.env_info",
    "naviertwin.utils.units",
    "naviertwin.utils.atomic_io",
    "naviertwin.utils.seeding",
    "naviertwin.core.digital_twin.manifest",
    "naviertwin.utils.rate_limit",
    "naviertwin.utils.event_bus",
    "naviertwin.utils.progress",
]


class TestRound100Milestone:
    @pytest.mark.parametrize("modname", MODULES)
    def test_importable(self, modname: str) -> None:
        import importlib

        importlib.import_module(modname)

    def test_end_to_end_mini(self) -> None:
        """preprocess → sweep → normalize → augment → event → progress 결합."""
        import numpy as np

        from naviertwin.core.augmentation.noise import add_gaussian_noise
        from naviertwin.core.preprocessing.normalizer import Normalizer
        from naviertwin.core.sampling.param_sweep import generate_sweep
        from naviertwin.utils.event_bus import EventBus
        from naviertwin.utils.progress import ProgressTracker
        from naviertwin.utils.seeding import set_global_seed

        set_global_seed(0)
        bus = EventBus()
        seen = []
        bus.subscribe("step", seen.append)

        tracker = ProgressTracker(total=3)
        pts = generate_sweep([(0, 1), (0, 1)], n_points=10, kind="lhs")
        bus.publish("step", {"name": "sweep", "n": len(pts)})
        tracker.update()

        X = np.random.default_rng(0).standard_normal((20, 8))
        Y = Normalizer("standard").fit_transform(X)
        bus.publish("step", {"name": "normalize", "std": float(Y.std())})
        tracker.update()

        Yn = add_gaussian_noise(Y, sigma=0.01, seed=0)
        bus.publish("step", {"name": "augment", "delta": float((Yn - Y).std())})
        tracker.update()

        assert len(seen) == 3
        assert tracker.fraction == 1.0
