"""Round 632 — surrogate certification: NRMSE/CV-RMSE, PICP/MPIW, calibration."""

from __future__ import annotations

import numpy as np
import pytest


class TestRMSE:
    def test_perfect_zero(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import rmse

        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import rmse

        yt = np.array([1.0, 2.0, 3.0])
        yp = np.array([2.0, 3.0, 4.0])
        # 모두 1 차이 → RMSE = 1
        np.testing.assert_allclose(rmse(yt, yp), 1.0)

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import rmse

        with pytest.raises(ValueError, match="shape"):
            rmse(np.zeros(5), np.zeros(4))


class TestNormalizedRMSE:
    def test_range_normalization(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import normalized_rmse

        yt = np.array([0.0, 5.0, 10.0])
        yp = np.array([1.0, 6.0, 9.0])
        # RMSE = 1, range = 10 → NRMSE = 0.1
        nrmse = normalized_rmse(yt, yp, norm="range")
        np.testing.assert_allclose(nrmse, 0.1, atol=1e-10)

    def test_mean_normalization(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import normalized_rmse

        yt = np.array([10.0, 20.0, 30.0])
        yp = np.array([11.0, 19.0, 31.0])
        nrmse = normalized_rmse(yt, yp, norm="mean")
        # RMSE = 1, mean = 20 → CV-RMSE = 0.05
        np.testing.assert_allclose(nrmse, 0.05, atol=1e-10)

    def test_iqr_norm(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import normalized_rmse

        yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        yp = yt + 0.1
        nrmse = normalized_rmse(yt, yp, norm="iqr")
        assert nrmse > 0

    def test_std_norm(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import normalized_rmse

        yt = np.array([1.0, 2.0, 3.0])
        yp = yt + 0.5
        nrmse = normalized_rmse(yt, yp, norm="std")
        assert nrmse > 0

    def test_invalid_norm(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import normalized_rmse

        with pytest.raises(ValueError, match="norm"):
            normalized_rmse(np.array([1.0]), np.array([1.0]), norm="bogus")

    def test_zero_scale_raises(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import normalized_rmse

        yt = np.zeros(5)
        yp = np.array([0.1, 0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="zero"):
            normalized_rmse(yt, yp, norm="mean")


class TestCVRMSE:
    def test_basic(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import cv_rmse

        yt = np.array([10.0, 20.0])
        yp = np.array([11.0, 19.0])
        v = cv_rmse(yt, yp)
        assert v > 0


class TestPICP:
    def test_full_coverage(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import picp

        yt = np.array([1.0, 2.0, 3.0])
        yl = yt - 1.0
        yu = yt + 1.0
        assert picp(yt, yl, yu) == 1.0

    def test_no_coverage(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import picp

        yt = np.array([10.0, 20.0])
        yl = np.array([1.0, 2.0])
        yu = np.array([3.0, 4.0])
        assert picp(yt, yl, yu) == 0.0

    def test_partial(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import picp

        yt = np.array([1.0, 2.0, 3.0, 4.0])
        yl = np.array([0.0, 0.0, 0.0, 5.0])
        yu = np.array([2.0, 1.5, 4.0, 6.0])
        # in: [0,2] yes, [0,1.5] no (2>1.5), [0,4] yes, [5,6] no
        assert picp(yt, yl, yu) == 0.5

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import picp

        with pytest.raises(ValueError, match="shape"):
            picp(np.zeros(3), np.zeros(3), np.zeros(4))


class TestMPIW:
    def test_basic(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import mpiw

        yl = np.array([0.0, 1.0])
        yu = np.array([2.0, 3.0])
        np.testing.assert_allclose(mpiw(yl, yu), 2.0)

    def test_normalized(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import mpiw

        yl = np.array([0.0, 1.0])
        yu = np.array([2.0, 3.0])
        ref = np.array([0.0, 10.0])  # range = 10
        np.testing.assert_allclose(mpiw(yl, yu, normalize_by_range=ref), 0.2)


class TestCWC:
    def test_above_target_no_penalty(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import (
            coverage_width_criterion,
        )

        yt = np.linspace(0, 10, 20)
        yl = yt - 1.0
        yu = yt + 1.0
        # 100% coverage, target 95% → no penalty
        cwc = coverage_width_criterion(yt, yl, yu, target_coverage=0.95)
        assert cwc > 0

    def test_below_target_penalty(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import (
            coverage_width_criterion,
        )

        yt = np.linspace(0, 10, 20)
        yl = yt + 5.0  # 모두 위
        yu = yt + 6.0
        cwc = coverage_width_criterion(yt, yl, yu, target_coverage=0.95)
        assert cwc > 0


class TestCalibrationCurve:
    def test_well_calibrated(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import (
            calibration_curve,
        )

        rng = np.random.default_rng(0)
        n = 1000
        yt = rng.standard_normal(n)
        ym = np.zeros(n)
        ys = np.ones(n)
        pred, obs = calibration_curve(yt, ym, ys, n_quantiles=11)
        # 정확히 정규분포 → calibration 잘 됨
        assert np.max(np.abs(pred - obs)) < 0.1

    def test_shape_mismatch(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import (
            calibration_curve,
        )

        with pytest.raises(ValueError, match="shape"):
            calibration_curve(np.zeros(10), np.zeros(10), np.zeros(8))

    def test_negative_std_raises(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import (
            calibration_curve,
        )

        with pytest.raises(ValueError, match="non-negative"):
            calibration_curve(
                np.zeros(5), np.zeros(5), np.array([1.0, 1.0, -1.0, 1.0, 1.0]),
            )


class TestECE:
    def test_well_calibrated_low(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import (
            expected_calibration_error,
        )

        rng = np.random.default_rng(1)
        yt = rng.standard_normal(2000)
        ece = expected_calibration_error(yt, np.zeros(2000), np.ones(2000))
        assert ece < 0.05

    def test_overconfident_higher(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import (
            expected_calibration_error,
        )

        rng = np.random.default_rng(2)
        yt = rng.standard_normal(1000)
        # σ를 너무 작게 → overconfident
        ece_bad = expected_calibration_error(yt, np.zeros(1000), 0.1 * np.ones(1000))
        ece_good = expected_calibration_error(yt, np.zeros(1000), np.ones(1000))
        assert ece_bad > ece_good


class TestR2:
    def test_perfect(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import r2_score

        y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(r2_score(y, y), 1.0)

    def test_mean_predictor_zero(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import r2_score

        y = np.array([1.0, 2.0, 3.0])
        yp = np.full_like(y, y.mean())
        np.testing.assert_allclose(r2_score(y, yp), 0.0)

    def test_constant_y_returns_nan(self) -> None:
        import math

        from naviertwin.core.surrogate.certification_metrics import r2_score

        r2 = r2_score(np.ones(5), np.ones(5))
        assert math.isnan(r2)

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.surrogate.certification_metrics import r2_score

        with pytest.raises(ValueError, match="shape"):
            r2_score(np.zeros(5), np.zeros(4))
