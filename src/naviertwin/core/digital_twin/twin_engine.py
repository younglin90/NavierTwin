"""디지털 트윈 예측 엔진.

POD/RandomizedPOD + RBF/Kriging 조합으로
파라미터 → POD 계수 → 유동장 재구성 파이프라인을 구현한다.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.dimensionality_reduction.base import BaseReducer
from naviertwin.core.surrogate.base import BaseSurrogate
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _build_reducer(reducer_type: str, n_modes: int) -> BaseReducer:
    """reducer_type 문자열로 차원 축소기를 생성한다.

    Args:
        reducer_type: "pod" | "randomized_pod".
        n_modes: 보존할 모드 수.

    Returns:
        BaseReducer 구체 인스턴스.

    Raises:
        ValueError: 지원되지 않는 reducer_type인 경우.
    """
    if reducer_type == "pod":
        from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD

        return SnapshotPOD(n_modes=n_modes)
    elif reducer_type == "randomized_pod":
        from naviertwin.core.dimensionality_reduction.linear.randomized_svd import RandomizedPOD

        return RandomizedPOD(n_modes=n_modes)
    else:
        raise ValueError(
            f"지원되지 않는 reducer_type: '{reducer_type}'. "
            f"지원 목록: ['pod', 'randomized_pod']"
        )


def _build_surrogate(surrogate_type: str) -> BaseSurrogate:
    """surrogate_type 문자열로 서로게이트 모델을 생성한다.

    Args:
        surrogate_type: "kriging" | "rbf".

    Returns:
        BaseSurrogate 구체 인스턴스.

    Raises:
        ValueError: 지원되지 않는 surrogate_type인 경우.
    """
    if surrogate_type == "kriging":
        from naviertwin.core.surrogate.kriging_surrogate import KrigingSurrogate

        return KrigingSurrogate()
    elif surrogate_type == "rbf":
        from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate

        return RBFSurrogate()
    else:
        raise ValueError(
            f"지원되지 않는 surrogate_type: '{surrogate_type}'. "
            f"지원 목록: ['kriging', 'rbf']"
        )


class TwinEngine:
    """파라미터 → 유동장 예측 파이프라인.

    POD/RandomizedPOD + RBF/Kriging 조합으로
    새로운 파라미터에서 유동장을 실시간 예측한다.

    파이프라인 구조:
        fit 시:
            1. reducer.fit(snapshots)          — POD 모드 학습
            2. coeffs = reducer.encode(snapshots)  — 계수 추출
            3. surrogate.fit(params, coeffs)   — 파라미터 → 계수 매핑 학습

        predict 시:
            1. coeffs = surrogate.predict(params)  — 새 파라미터의 계수 예측
            2. field = reducer.decode(coeffs)       — 계수 → 유동장 재구성

    Attributes:
        reducer: 차원 축소기 (BaseReducer 인스턴스).
        surrogate: 서로게이트 모델 (BaseSurrogate 인스턴스).

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.digital_twin.twin_engine import TwinEngine
        >>> rng = np.random.default_rng(0)
        >>> snapshots = rng.standard_normal((100, 30))  # (n_features, n_samples)
        >>> params = rng.uniform(0, 1, (30, 2))         # (n_samples, n_params)
        >>> engine = TwinEngine(reducer_type="pod", surrogate_type="rbf", n_modes=5)
        >>> engine.fit(snapshots, params)
        >>> field = engine.predict(params[:3])
    """

    def __init__(
        self,
        reducer_type: str = "pod",
        surrogate_type: str = "kriging",
        n_modes: int = 10,
    ) -> None:
        """초기화.

        Args:
            reducer_type: 차원 축소 방법. "pod" | "randomized_pod".
            surrogate_type: 서로게이트 모델 종류. "kriging" | "rbf".
            n_modes: 보존할 POD 모드 수.
        """
        self.reducer_type = reducer_type
        self.surrogate_type = surrogate_type
        self.n_modes = n_modes

        self.reducer: BaseReducer = _build_reducer(reducer_type, n_modes)
        self.surrogate: BaseSurrogate = _build_surrogate(surrogate_type)

        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def fit(
        self,
        snapshots: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> None:
        """스냅샷과 대응 파라미터로 파이프라인을 학습한다.

        Args:
            snapshots: 스냅샷 행렬. shape = (n_features, n_samples).
            params: 각 스냅샷에 대응하는 파라미터. shape = (n_samples, n_params).

        Raises:
            ValueError: snapshots와 params의 샘플 수가 일치하지 않는 경우.
        """
        if snapshots.ndim != 2:
            raise ValueError(
                f"snapshots는 2D여야 합니다. 현재 shape: {snapshots.shape}"
            )
        if params.ndim != 2:
            raise ValueError(
                f"params는 2D여야 합니다. 현재 shape: {params.shape}"
            )

        n_features, n_samples = snapshots.shape
        if params.shape[0] != n_samples:
            raise ValueError(
                f"snapshots의 샘플 수({n_samples})와 "
                f"params의 샘플 수({params.shape[0]})가 일치하지 않습니다."
            )

        logger.info(
            "TwinEngine.fit 시작: n_features=%d, n_samples=%d, n_params=%d",
            n_features,
            n_samples,
            params.shape[1],
        )

        # Step 1: 차원 축소기 학습
        logger.debug("Step 1: %s.fit(snapshots) 수행 중...", self.reducer.__class__.__name__)
        self.reducer.fit(snapshots)

        # Step 2: 스냅샷을 POD 계수로 인코딩
        logger.debug("Step 2: encode(snapshots) 수행 중...")
        coeffs = self.reducer.encode(snapshots)  # (n_samples, n_modes)

        # Step 3: 서로게이트 모델 학습 (params → coeffs)
        logger.debug(
            "Step 3: %s.fit(params, coeffs) 수행 중...",
            self.surrogate.__class__.__name__,
        )
        self.surrogate.fit(params, coeffs)

        self._is_fitted = True
        logger.info(
            "TwinEngine 학습 완료: reducer=%s, surrogate=%s, n_modes=%d",
            self.reducer.__class__.__name__,
            self.surrogate.__class__.__name__,
            self.reducer.n_components,
        )

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------

    def predict(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """새로운 파라미터에서 유동장을 예측한다.

        Args:
            params: 새로운 파라미터. shape = (n_samples, n_params) 또는 (n_params,).

        Returns:
            예측된 유동장.
                - 단일 샘플: shape = (n_features,)
                - 다중 샘플: shape = (n_features, n_samples)

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        self._check_fitted()

        is_single = params.ndim == 1
        if is_single:
            params = params[np.newaxis, :]

        # Step 1: 파라미터 → POD 계수 예측
        coeffs = self.surrogate.predict(params)  # (n_samples, n_modes)

        # surrogate가 1D 반환할 경우 (단일 출력) 2D로 변환
        if coeffs.ndim == 1:
            coeffs = coeffs[np.newaxis, :]

        # Step 2: POD 계수 → 유동장 재구성
        field = self.reducer.decode(coeffs)  # (n_features, n_samples)

        if is_single:
            return field[:, 0]  # (n_features,)
        return field  # (n_features, n_samples)

    # ------------------------------------------------------------------
    # 저장 / 로드
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """엔진을 pickle 파일로 저장한다.

        Args:
            path: 저장할 파일 경로. 부모 디렉토리가 없으면 자동 생성한다.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("TwinEngine 저장 완료: %s", path)

    @classmethod
    def load(cls, path: Path) -> "TwinEngine":
        """저장된 엔진을 로드한다.

        Args:
            path: 로드할 pickle 파일 경로.

        Returns:
            로드된 TwinEngine 인스턴스.

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"저장된 엔진 파일을 찾을 수 없습니다: {path}")

        with path.open("rb") as f:
            engine = pickle.load(f)

        logger.info("TwinEngine 로드 완료: %s", path)
        return engine

    # ------------------------------------------------------------------
    # 프로퍼티 및 유틸리티
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """fit() 완료 여부.

        Returns:
            fit()이 성공적으로 완료되었으면 True.
        """
        return self._is_fitted

    def _check_fitted(self) -> None:
        """fit() 완료 여부를 확인한다.

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "TwinEngine.predict() 전에 fit()을 먼저 호출해야 합니다."
            )

    def get_params(self) -> dict[str, Any]:
        """엔진 설정 파라미터를 반환한다.

        Returns:
            {"reducer_type": ..., "surrogate_type": ..., "n_modes": ...}
        """
        return {
            "reducer_type": self.reducer_type,
            "surrogate_type": self.surrogate_type,
            "n_modes": self.n_modes,
        }

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"TwinEngine("
            f"reducer={self.reducer.__class__.__name__}, "
            f"surrogate={self.surrogate.__class__.__name__}, "
            f"n_modes={self.n_modes}, "
            f"status={status})"
        )
