"""Round 84 — 데이터 증강 유틸."""

from __future__ import annotations

import numpy as np
import pytest


class TestNoise:
    def test_gaussian_reproducible(self) -> None:
        from naviertwin.core.augmentation.noise import add_gaussian_noise

        X = np.zeros((20, 5))
        a = add_gaussian_noise(X, sigma=0.1, seed=0)
        b = add_gaussian_noise(X, sigma=0.1, seed=0)
        assert np.allclose(a, b)
        assert 0.05 < a.std() < 0.2

    def test_gaussian_relative(self) -> None:
        from naviertwin.core.augmentation.noise import add_gaussian_noise

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 5)) * 10.0  # std ≈ 10
        Y = add_gaussian_noise(X, sigma=0.05, relative=True, seed=0)
        # 노이즈 std ≈ 0.05 * 10 = 0.5
        assert 0.2 < (Y - X).std() < 1.0

    def test_uniform(self) -> None:
        from naviertwin.core.augmentation.noise import add_uniform_noise

        X = np.zeros((100, 3))
        Y = add_uniform_noise(X, amplitude=0.5, seed=0)
        assert np.max(np.abs(Y - X)) <= 0.5

    def test_dropout(self) -> None:
        from naviertwin.core.augmentation.noise import random_dropout

        X = np.ones((1000,))
        Y = random_dropout(X, drop_rate=0.3, seed=0)
        zeros = np.sum(Y == 0.0)
        assert 200 < zeros < 400

    def test_dropout_invalid(self) -> None:
        from naviertwin.core.augmentation.noise import random_dropout

        with pytest.raises(ValueError):
            random_dropout(np.zeros(5), drop_rate=1.5)

    def test_augment_batch(self) -> None:
        from naviertwin.core.augmentation.noise import augment_batch

        X = np.ones((10, 4))
        Y = augment_batch(X, n_copies=3, sigma=0.01, seed=0)
        assert Y.shape == (10, 16)
        # 첫 4열은 원본
        assert np.allclose(Y[:, :4], X)
