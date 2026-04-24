"""Round 203 — image metrics."""

from __future__ import annotations

import numpy as np


class TestImg:
    def test_psnr_identical(self) -> None:
        from naviertwin.core.validation.image_metrics import psnr

        a = np.ones((8, 8))
        assert psnr(a, a) == float("inf")

    def test_psnr_small_diff(self) -> None:
        from naviertwin.core.validation.image_metrics import psnr

        a = np.ones((8, 8))
        b = a + 0.001
        assert psnr(a, b, data_range=1.0) > 50

    def test_ssim_identical(self) -> None:
        from naviertwin.core.validation.image_metrics import ssim

        rng = np.random.default_rng(0)
        a = rng.uniform(0, 1, (16, 16))
        assert abs(ssim(a, a) - 1.0) < 1e-6

    def test_ssim_different(self) -> None:
        from naviertwin.core.validation.image_metrics import ssim

        rng = np.random.default_rng(0)
        a = rng.uniform(0, 1, (16, 16))
        b = rng.uniform(0, 1, (16, 16))
        assert ssim(a, b) < 0.3

    def test_nrmse(self) -> None:
        from naviertwin.core.validation.image_metrics import nrmse

        a = np.array([[0.0, 1.0], [0.0, 1.0]])
        b = a + 0.1
        assert abs(nrmse(a, b) - 0.1) < 1e-10
