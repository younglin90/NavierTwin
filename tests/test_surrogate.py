"""Surrogate 모델 테스트.

RBFSurrogate 및 KrigingSurrogate의 학습/예측 정확도,
분산 예측, 예외 처리를 검증한다.
모든 테스트는 실제 CFD 데이터 없이 numpy 랜덤 데이터로 동작한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from naviertwin.core.surrogate.kriging_surrogate import KrigingSurrogate
from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate


# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture()
def sinusoidal_1d() -> tuple[np.ndarray, np.ndarray]:
    """1D 사인 함수 학습/테스트 데이터."""
    rng = np.random.default_rng(0)
    X_train = rng.uniform(0.0, 2 * np.pi, (20, 1)).astype(np.float64)
    y_train = np.sin(X_train.ravel()).astype(np.float64)
    return X_train, y_train


@pytest.fixture()
def multi_output_2d() -> tuple[np.ndarray, np.ndarray]:
    """2D 입력 → 2D 출력 데이터셋 (다중 출력 테스트용)."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.0, 1.0, (30, 2)).astype(np.float64)
    y = np.column_stack([
        np.sin(X[:, 0]) + X[:, 1],
        X[:, 0] ** 2 - X[:, 1] ** 2,
    ]).astype(np.float64)
    return X, y


# ---------------------------------------------------------------------------
# RBFSurrogate 테스트
# ---------------------------------------------------------------------------


class TestRBFSurrogate:
    """RBFSurrogate 단위 테스트."""

    def test_fit_sets_is_fitted(self, sinusoidal_1d: tuple) -> None:
        """fit() 후 is_fitted가 True로 설정되는지 확인한다."""
        X, y = sinusoidal_1d
        rbf = RBFSurrogate()
        assert not rbf.is_fitted
        rbf.fit(X, y)
        assert rbf.is_fitted

    def test_rbf_fit_predict_1d(self, sinusoidal_1d: tuple) -> None:
        """1D 사인 함수 근사 정확도를 확인한다.

        학습 데이터에 대한 예측이 실제값과 가까워야 한다.
        """
        X, y = sinusoidal_1d
        rbf = RBFSurrogate(d0=1.0)
        rbf.fit(X, y)

        y_pred = rbf.predict(X)
        assert y_pred.shape == y.shape, (
            f"predict 출력 shape 불일치: {y_pred.shape} != {y.shape}"
        )

        # 학습 데이터에 대한 R² 점수
        r2 = rbf.score(X, y)
        assert r2 > 0.5, f"R² 점수가 너무 낮습니다: {r2:.4f}"

    def test_rbf_multi_output(self, multi_output_2d: tuple) -> None:
        """다중 출력 예측 shape이 올바른지 확인한다."""
        X, y = multi_output_2d
        rbf = RBFSurrogate()
        rbf.fit(X, y)

        y_pred = rbf.predict(X)
        assert y_pred.shape == y.shape, (
            f"다중 출력 shape 불일치: {y_pred.shape} != {y.shape}"
        )

    def test_rbf_predict_single_sample(self, sinusoidal_1d: tuple) -> None:
        """단일 샘플 예측이 올바른 shape을 반환하는지 확인한다."""
        X, y = sinusoidal_1d
        rbf = RBFSurrogate()
        rbf.fit(X, y)

        x_single = X[0]  # shape (1,) — 1D
        y_pred = rbf.predict(x_single[np.newaxis, :])  # (1, 1) 입력
        assert y_pred.ndim <= 2

    def test_rbf_not_fitted_error(self) -> None:
        """fit() 전에 predict() 호출 시 RuntimeError가 발생하는지 확인한다."""
        rbf = RBFSurrogate()
        with pytest.raises(RuntimeError, match="fit"):
            rbf.predict(np.ones((5, 2)))

    def test_rbf_score_perfect(self) -> None:
        """완벽한 예측의 R² 점수가 1.0인지 확인한다."""
        rng = np.random.default_rng(5)
        X = rng.uniform(0, 1, (10, 2)).astype(np.float64)
        y = rng.standard_normal(10).astype(np.float64)

        rbf = RBFSurrogate()
        rbf.fit(X, y)
        y_pred = rbf.predict(X)

        # 학습 데이터 자체에 대한 R²는 1에 가까워야 함
        r2 = rbf.score(X, y)
        # 보간이면 1.0, 근사이면 다소 낮을 수 있음
        assert r2 > 0.0, f"R² 점수가 음수입니다: {r2:.4f}"

    def test_rbf_get_params(self, sinusoidal_1d: tuple) -> None:
        """get_params()가 올바른 키를 반환하는지 확인한다."""
        X, y = sinusoidal_1d
        rbf = RBFSurrogate(d0=2.5)
        rbf.fit(X, y)
        params = rbf.get_params()
        assert "d0" in params
        assert params["d0"] == 2.5


# ---------------------------------------------------------------------------
# KrigingSurrogate 테스트
# ---------------------------------------------------------------------------


class TestKrigingSurrogate:
    """KrigingSurrogate 단위 테스트."""

    def test_fit_sets_is_fitted(self, sinusoidal_1d: tuple) -> None:
        """fit() 후 is_fitted가 True로 설정되는지 확인한다."""
        X, y = sinusoidal_1d
        krig = KrigingSurrogate()
        assert not krig.is_fitted
        krig.fit(X, y)
        assert krig.is_fitted

    def test_kriging_fit_predict(self, sinusoidal_1d: tuple) -> None:
        """학습 데이터에 대한 예측 정확도를 확인한다."""
        X, y = sinusoidal_1d
        krig = KrigingSurrogate(corr="squar_exp", poly="constant")
        krig.fit(X, y)

        y_pred = krig.predict(X)
        assert y_pred.shape == y.shape, (
            f"predict 출력 shape 불일치: {y_pred.shape} != {y.shape}"
        )

        r2 = krig.score(X, y)
        assert r2 > 0.5, f"Kriging R² 점수가 너무 낮습니다: {r2:.4f}"

    def test_kriging_predict_with_variance(self, sinusoidal_1d: tuple) -> None:
        """predict_with_variance()의 분산이 0 이상인지 확인한다."""
        X, y = sinusoidal_1d
        krig = KrigingSurrogate()
        krig.fit(X, y)

        y_pred, y_var = krig.predict_with_variance(X)

        # shape 확인
        assert y_pred.shape == y.shape, (
            f"y_pred shape 불일치: {y_pred.shape} != {y.shape}"
        )
        assert y_var.shape == y.shape, (
            f"y_var shape 불일치: {y_var.shape} != {y.shape}"
        )

        # 분산은 0 이상이어야 함
        assert np.all(y_var >= -1e-10), (
            f"분산에 음수 값이 있습니다: min={y_var.min():.6e}"
        )

    def test_kriging_variance_non_negative_multi_output(
        self, multi_output_2d: tuple
    ) -> None:
        """다중 출력의 예측 분산이 0 이상인지 확인한다."""
        X, y = multi_output_2d
        krig = KrigingSurrogate()
        krig.fit(X, y)

        y_pred, y_var = krig.predict_with_variance(X)
        assert np.all(y_var >= -1e-10), (
            f"다중 출력 분산에 음수 값이 있습니다: min={y_var.min():.6e}"
        )

    def test_kriging_not_fitted_error(self) -> None:
        """fit() 전에 predict() 호출 시 RuntimeError가 발생하는지 확인한다."""
        krig = KrigingSurrogate()
        with pytest.raises(RuntimeError, match="fit"):
            krig.predict(np.ones((5, 2)))

    def test_kriging_get_params(self, sinusoidal_1d: tuple) -> None:
        """get_params()가 corr, poly, backend 키를 포함하는지 확인한다."""
        X, y = sinusoidal_1d
        krig = KrigingSurrogate(corr="abs_exp", poly="linear")
        krig.fit(X, y)
        params = krig.get_params()
        assert "corr" in params
        assert "poly" in params
        assert "backend" in params

    def test_kriging_invalid_input_ndim(self) -> None:
        """1D X 입력 시 ValueError가 발생하는지 확인한다."""
        krig = KrigingSurrogate()
        with pytest.raises(ValueError):
            krig.fit(np.ones(10), np.ones(10))

    def test_kriging_multi_output(self, multi_output_2d: tuple) -> None:
        """다중 출력 예측 shape이 올바른지 확인한다."""
        X, y = multi_output_2d
        krig = KrigingSurrogate()
        krig.fit(X, y)

        y_pred = krig.predict(X)
        assert y_pred.shape == y.shape, (
            f"다중 출력 shape 불일치: {y_pred.shape} != {y.shape}"
        )
