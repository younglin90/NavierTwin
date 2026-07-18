"""Round 641 — multi-output regression metrics."""

from __future__ import annotations

import numpy as np
import pytest


class TestChannelRMSE:
    def test_perfect_zero(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import channel_rmse

        y = np.random.default_rng(0).standard_normal((50, 3))
        rmse = channel_rmse(y, y)
        np.testing.assert_array_equal(rmse, np.zeros(3))

    def test_known_value(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import channel_rmse

        yt = np.array([[1.0, 1.0], [2.0, 2.0]])
        yp = np.array([[2.0, 1.0], [3.0, 2.0]])
        # 채널 0: 차이 1 → RMSE=1; 채널 1: 차이 0 → 0
        rmse = channel_rmse(yt, yp)
        np.testing.assert_allclose(rmse, [1.0, 0.0])

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import channel_rmse

        with pytest.raises(ValueError, match="shape"):
            channel_rmse(np.zeros((5, 3)), np.zeros((5, 4)))

    def test_1d_raises(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import channel_rmse

        with pytest.raises(ValueError, match="2D"):
            channel_rmse(np.zeros(10), np.zeros(10))


class TestRelativeError:
    def test_basic(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import (
            channel_relative_error,
        )

        yt = np.array([[1.0, 10.0], [2.0, 20.0]])
        yp = yt + 0.1
        # 0.1 차이 / 작은 채널 → 큰 상대 오차
        rel = channel_relative_error(yt, yp)
        assert rel[0] > rel[1]


class TestAggregatedRMSE:
    def test_uniform_weight(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import aggregated_rmse

        yt = np.array([[1.0, 0.0], [2.0, 0.0]])
        yp = np.array([[2.0, 0.0], [3.0, 0.0]])
        # 채널 0 RMSE=1, 채널 1 RMSE=0 → agg = √(0.5)
        agg = aggregated_rmse(yt, yp)
        np.testing.assert_allclose(agg, np.sqrt(0.5))

    def test_explicit_weights(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import aggregated_rmse

        yt = np.array([[1.0, 0.0], [2.0, 0.0]])
        yp = np.array([[2.0, 0.0], [3.0, 0.0]])
        # 가중치를 채널 1로 → agg = 0
        agg = aggregated_rmse(yt, yp, weights=np.array([0.0, 1.0]))
        np.testing.assert_allclose(agg, 0.0)

    def test_invalid_weight_shape(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import aggregated_rmse

        with pytest.raises(ValueError, match="weights"):
            aggregated_rmse(
                np.zeros((5, 3)), np.zeros((5, 3)), weights=np.array([1.0]),
            )

    def test_zero_weights_raises(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import aggregated_rmse

        with pytest.raises(ValueError, match="zero"):
            aggregated_rmse(
                np.zeros((5, 2)), np.zeros((5, 2)), weights=np.zeros(2),
            )


class TestMultiOutputR2:
    def test_perfect_unity(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import multi_output_r2

        rng = np.random.default_rng(0)
        y = rng.standard_normal((50, 3))
        np.testing.assert_allclose(multi_output_r2(y, y, average="uniform"), 1.0)

    def test_raw_returns_array(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import multi_output_r2

        rng = np.random.default_rng(1)
        y = rng.standard_normal((50, 3))
        r2 = multi_output_r2(y, y, average="raw")
        assert r2.shape == (3,)

    def test_variance_weighted(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import multi_output_r2

        rng = np.random.default_rng(2)
        y = rng.standard_normal((50, 3))
        # 채널 1만 큰 분산
        y[:, 1] *= 10.0
        r2 = multi_output_r2(y, y, average="variance_weighted")
        np.testing.assert_allclose(r2, 1.0)

    def test_constant_y_returns_nan_in_raw(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import multi_output_r2

        y = np.ones((10, 3))
        r2 = multi_output_r2(y, y, average="raw")
        assert np.all(np.isnan(r2))

    def test_invalid_average(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import multi_output_r2

        with pytest.raises(ValueError, match="average"):
            multi_output_r2(np.zeros((5, 2)), np.zeros((5, 2)), average="bogus")


class TestCrossChannelCorrelation:
    def test_perfect_correlation(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import (
            cross_channel_correlation,
        )

        rng = np.random.default_rng(3)
        y = rng.standard_normal((100, 3))
        rho = cross_channel_correlation(y, y)
        np.testing.assert_allclose(rho, 1.0, atol=1e-10)

    def test_negative_correlation(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import (
            cross_channel_correlation,
        )

        rng = np.random.default_rng(4)
        yt = rng.standard_normal((100, 1))
        yp = -yt
        rho = cross_channel_correlation(yt, yp)
        np.testing.assert_allclose(rho, -1.0, atol=1e-10)


class TestTopKWorst:
    def test_basic(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import (
            top_k_worst_channels,
        )

        # 채널 2가 가장 나쁨
        yt = np.zeros((10, 4))
        yp = np.zeros((10, 4))
        yp[:, 2] = 5.0  # 큰 오차
        idx = top_k_worst_channels(yt, yp, k=2)
        assert idx[0] == 2

    def test_invalid_k(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import (
            top_k_worst_channels,
        )

        with pytest.raises(ValueError, match="k"):
            top_k_worst_channels(np.zeros((5, 3)), np.zeros((5, 3)), k=10)


class TestPerSampleErrorNorm:
    def test_zero_for_identical(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import (
            per_sample_error_norm,
        )

        y = np.ones((10, 3))
        norms = per_sample_error_norm(y, y)
        np.testing.assert_array_equal(norms, np.zeros(10))

    def test_known_norm(self) -> None:
        from naviertwin.core.validation.multi_output_metrics import (
            per_sample_error_norm,
        )

        yt = np.zeros((2, 2))
        yp = np.array([[3.0, 4.0], [0.0, 0.0]])
        # 표본 0: √(9+16) = 5; 표본 1: 0
        norms = per_sample_error_norm(yt, yp)
        np.testing.assert_allclose(norms, [5.0, 0.0])
