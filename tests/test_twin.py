"""디지털 트윈 엔진 및 검증 지표 테스트.

TwinEngine의 fit/predict 파이프라인, 저장/로드 일관성,
validation/metrics 계산 정확도를 검증한다.
모든 테스트는 실제 CFD 데이터 없이 numpy 랜덤 데이터로 동작한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from naviertwin.core.digital_twin.twin_engine import TwinEngine
from naviertwin.core.validation.metrics import (
    compute_all_metrics,
    max_error,
    r2_score,
    relative_l2_error,
    rmse,
)


# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture()
def low_rank_problem() -> tuple[np.ndarray, np.ndarray]:
    """저차원 구조의 스냅샷 + 파라미터 데이터.

    Returns:
        (snapshots, params) 튜플.
            snapshots: (n_features=80, n_samples=25)
            params: (n_samples=25, n_params=2)
    """
    rng = np.random.default_rng(2025)
    n_features = 80
    n_samples = 25
    n_params = 2
    n_modes = 4

    # 파라미터 생성
    params = rng.uniform(0.0, 1.0, (n_samples, n_params)).astype(np.float64)

    # 파라미터에 의존하는 저차원 스냅샷 생성
    basis = rng.standard_normal((n_features, n_modes))
    coeffs_true = np.column_stack([
        np.sin(np.pi * params[:, 0]),
        np.cos(np.pi * params[:, 1]),
        params[:, 0] * params[:, 1],
        params[:, 0] ** 2,
    ])  # (n_samples, n_modes)
    snapshots = (basis @ coeffs_true.T).astype(np.float64)  # (n_features, n_samples)

    return snapshots, params


# ---------------------------------------------------------------------------
# TwinEngine 테스트
# ---------------------------------------------------------------------------


class TestTwinEngine:
    """TwinEngine 통합 테스트."""

    @pytest.mark.parametrize(
        "reducer_type,surrogate_type",
        [
            ("pod", "rbf"),
            ("pod", "kriging"),
            ("randomized_pod", "rbf"),
        ],
    )
    def test_twin_engine_fit_predict(
        self,
        low_rank_problem: tuple,
        reducer_type: str,
        surrogate_type: str,
    ) -> None:
        """파라미터 → 유동장 예측 왕복 테스트.

        학습 데이터에 대한 예측이 충분한 정확도를 가져야 한다.
        """
        snapshots, params = low_rank_problem
        n_features, n_samples = snapshots.shape

        engine = TwinEngine(
            reducer_type=reducer_type,
            surrogate_type=surrogate_type,
            n_modes=4,
        )
        assert not engine.is_fitted

        engine.fit(snapshots, params)
        assert engine.is_fitted

        # 단일 샘플 예측
        field_single = engine.predict(params[0])
        assert field_single.shape == (n_features,), (
            f"단일 예측 shape 불일치: {field_single.shape} != ({n_features},)"
        )

        # 다중 샘플 예측
        field_multi = engine.predict(params[:5])
        assert field_multi.shape == (n_features, 5), (
            f"다중 예측 shape 불일치: {field_multi.shape} != ({n_features}, 5)"
        )

    def test_twin_engine_accuracy(self, low_rank_problem: tuple) -> None:
        """학습 데이터에 대한 예측 정확도가 충분한지 확인한다."""
        snapshots, params = low_rank_problem

        engine = TwinEngine(reducer_type="pod", surrogate_type="rbf", n_modes=4)
        engine.fit(snapshots, params)

        field_pred = engine.predict(params)  # (n_features, n_samples)

        rel_err = np.linalg.norm(snapshots - field_pred) / np.linalg.norm(snapshots)
        assert rel_err < 0.5, (
            f"TwinEngine 학습 데이터 상대 오차가 너무 큽니다: {rel_err:.4f}"
        )

    def test_twin_engine_not_fitted_error(self) -> None:
        """fit() 전에 predict() 호출 시 RuntimeError가 발생하는지 확인한다."""
        engine = TwinEngine()
        with pytest.raises(RuntimeError, match="fit"):
            engine.predict(np.ones((3, 2)))

    def test_twin_engine_invalid_snapshots_shape(
        self, low_rank_problem: tuple
    ) -> None:
        """1D snapshots 입력 시 ValueError가 발생하는지 확인한다."""
        _, params = low_rank_problem
        engine = TwinEngine(n_modes=4)
        with pytest.raises(ValueError):
            engine.fit(np.ones(50), params)

    def test_twin_engine_sample_mismatch(self, low_rank_problem: tuple) -> None:
        """snapshots와 params의 샘플 수 불일치 시 ValueError가 발생하는지 확인한다."""
        snapshots, params = low_rank_problem
        engine = TwinEngine(n_modes=4)
        wrong_params = params[:5]  # 샘플 수 불일치
        with pytest.raises(ValueError, match="샘플 수"):
            engine.fit(snapshots, wrong_params)

    def test_twin_engine_save_load(
        self, low_rank_problem: tuple, tmp_path: Path
    ) -> None:
        """저장/로드 후 동일한 예측 결과를 반환하는지 확인한다."""
        snapshots, params = low_rank_problem

        engine = TwinEngine(reducer_type="pod", surrogate_type="rbf", n_modes=4)
        engine.fit(snapshots, params)

        # 저장
        save_path = tmp_path / "engine.pkl"
        engine.save(save_path)
        assert save_path.exists(), "저장 파일이 생성되지 않았습니다."

        # 로드
        loaded_engine = TwinEngine.load(save_path)
        assert loaded_engine.is_fitted, "로드된 엔진이 fitted 상태가 아닙니다."

        # 동일한 예측 결과 비교
        test_params = params[:3]
        pred_original = engine.predict(test_params)
        pred_loaded = loaded_engine.predict(test_params)

        np.testing.assert_allclose(
            pred_original,
            pred_loaded,
            rtol=1e-10,
            err_msg="저장/로드 전후 예측 결과가 다릅니다.",
        )

    def test_twin_engine_load_nonexistent(self, tmp_path: Path) -> None:
        """존재하지 않는 파일 로드 시 FileNotFoundError가 발생하는지 확인한다."""
        with pytest.raises(FileNotFoundError):
            TwinEngine.load(tmp_path / "nonexistent.pkl")

    def test_twin_engine_repr(self, low_rank_problem: tuple) -> None:
        """repr() 출력이 올바른지 확인한다."""
        engine = TwinEngine(n_modes=4)
        r = repr(engine)
        assert "TwinEngine" in r
        assert "not fitted" in r

        snapshots, params = low_rank_problem
        engine.fit(snapshots, params)
        r_fitted = repr(engine)
        assert "fitted" in r_fitted

    def test_twin_engine_get_params(self) -> None:
        """get_params()가 올바른 키를 반환하는지 확인한다."""
        engine = TwinEngine(reducer_type="randomized_pod", surrogate_type="rbf", n_modes=8)
        params = engine.get_params()
        assert params["reducer_type"] == "randomized_pod"
        assert params["surrogate_type"] == "rbf"
        assert params["n_modes"] == 8

    def test_invalid_reducer_type(self) -> None:
        """지원되지 않는 reducer_type 시 ValueError가 발생하는지 확인한다."""
        with pytest.raises(ValueError, match="reducer_type"):
            TwinEngine(reducer_type="unknown_method")

    def test_invalid_surrogate_type(self) -> None:
        """지원되지 않는 surrogate_type 시 ValueError가 발생하는지 확인한다."""
        with pytest.raises(ValueError, match="surrogate_type"):
            TwinEngine(surrogate_type="unknown_surrogate")


# ---------------------------------------------------------------------------
# 검증 지표 테스트
# ---------------------------------------------------------------------------


class TestValidationMetrics:
    """validation/metrics.py 단위 테스트."""

    def test_rmse_perfect_prediction(self) -> None:
        """완벽한 예측의 RMSE가 0인지 확인한다."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert rmse(y, y) == pytest.approx(0.0, abs=1e-12)

    def test_rmse_known_value(self) -> None:
        """알려진 오차에 대한 RMSE 계산 정확도를 확인한다."""
        y_true = np.zeros(4)
        y_pred = np.array([1.0, -1.0, 1.0, -1.0])
        # RMSE = sqrt(mean([1,1,1,1])) = 1.0
        assert rmse(y_true, y_pred) == pytest.approx(1.0, rel=1e-6)

    def test_r2_score_perfect(self) -> None:
        """완벽한 예측의 R²가 1.0인지 확인한다."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r2_score(y, y) == pytest.approx(1.0, abs=1e-12)

    def test_r2_score_mean_prediction(self) -> None:
        """평균값 예측의 R²가 0.0인지 확인한다."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert r2_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_r2_score_constant_true(self) -> None:
        """y_true가 상수일 때 R²가 1.0(완벽) 또는 0.0(불일치)인지 확인한다."""
        y_const = np.ones(5)
        # 동일한 예측: R² = 1.0
        assert r2_score(y_const, y_const) == pytest.approx(1.0)
        # 다른 예측: R² = 0.0 (fallback)
        r2 = r2_score(y_const, y_const + 1.0)
        assert r2 == pytest.approx(0.0, abs=1e-10)

    def test_relative_l2_error_perfect(self) -> None:
        """완벽한 예측의 상대 L2 오차가 0인지 확인한다."""
        y = np.array([1.0, 2.0, 3.0])
        assert relative_l2_error(y, y) == pytest.approx(0.0, abs=1e-12)

    def test_relative_l2_error_known(self) -> None:
        """알려진 상대 L2 오차 계산 정확도를 확인한다."""
        y_true = np.ones(4)           # L2 norm = 2.0
        y_pred = np.ones(4) * 1.5    # diff = [0.5, 0.5, 0.5, 0.5], L2 norm = 1.0
        expected = 1.0 / 2.0
        assert relative_l2_error(y_true, y_pred) == pytest.approx(expected, rel=1e-6)

    def test_relative_l2_zero_denominator(self) -> None:
        """y_true가 영벡터일 때 절대 오차를 반환하는지 확인한다."""
        y_true = np.zeros(5)
        y_pred = np.ones(5)
        result = relative_l2_error(y_true, y_pred)
        # 분모가 0이므로 절대 L2 오차 = sqrt(5) 반환
        assert result == pytest.approx(float(np.linalg.norm(y_pred)), rel=1e-6)

    def test_max_error_perfect(self) -> None:
        """완벽한 예측의 최대 절대 오차가 0인지 확인한다."""
        y = np.array([1.0, 2.0, 3.0])
        assert max_error(y, y) == pytest.approx(0.0, abs=1e-12)

    def test_max_error_known(self) -> None:
        """알려진 최대 절대 오차 계산 정확도를 확인한다."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([0.1, 0.5, 0.3])
        assert max_error(y_true, y_pred) == pytest.approx(0.5, rel=1e-6)

    def test_validation_metrics(self) -> None:
        """compute_all_metrics()가 올바른 키와 값을 반환하는지 확인한다."""
        rng = np.random.default_rng(99)
        y_true = rng.standard_normal(100)
        y_pred = y_true + 0.1 * rng.standard_normal(100)

        metrics = compute_all_metrics(y_true, y_pred)

        # 키 확인
        assert set(metrics.keys()) == {"rmse", "r2", "relative_l2", "max_error"}, (
            f"반환된 키가 예상과 다릅니다: {set(metrics.keys())}"
        )

        # 값 타입 확인
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key}의 타입이 float가 아닙니다: {type(val)}"

        # 합리적인 범위 확인
        assert metrics["rmse"] >= 0.0
        assert metrics["relative_l2"] >= 0.0
        assert metrics["max_error"] >= 0.0
        assert metrics["r2"] <= 1.0

    def test_metrics_with_2d_arrays(self) -> None:
        """2D 배열 입력에 대한 지표 계산이 올바른지 확인한다."""
        rng = np.random.default_rng(7)
        y_true = rng.standard_normal((10, 5))
        y_pred = y_true + 0.05 * rng.standard_normal((10, 5))

        # 에러 없이 실행되어야 함
        result = compute_all_metrics(y_true, y_pred)
        assert result["rmse"] >= 0.0
        assert result["r2"] <= 1.0
