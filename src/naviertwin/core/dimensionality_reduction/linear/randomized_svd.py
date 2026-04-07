"""Randomized SVD 기반 고속 POD (대용량 스냅샷용).

sklearn.utils.extmath.randomized_svd 를 이용하여
O(n_features * n_modes) 복잡도로 근사 SVD를 수행한다.
sklearn 미설치 시 numpy.linalg.svd 로 폴백한다.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.dimensionality_reduction.base import BaseReducer
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class RandomizedPOD(BaseReducer):
    """Randomized SVD 기반 고속 POD.

    sklearn.utils.extmath.randomized_svd 를 사용하여 대용량
    스냅샷 행렬에서도 빠르게 POD 모드를 추출한다.

    sklearn 미설치 시 numpy.linalg.svd 로 자동 폴백한다.

    Attributes:
        n_modes: 보존할 모드 수.
        n_oversamples: 랜덤 오버샘플링 파라미터 (sklearn randomized_svd).
        random_state: 재현성을 위한 난수 시드.
        center: True이면 평균 필드를 제거한 뒤 SVD를 수행한다.
        modes_: shape (n_features, n_modes) POD 모드 (fit 후 설정).
        singular_values_: shape (n_modes,) 특이값 (fit 후 설정).
        mean_: shape (n_features,) 평균 필드 (fit 후 설정).
        energy_ratio_: shape (n_modes,) 누적 에너지 기여율 (fit 후 설정).

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.dimensionality_reduction.linear.randomized_svd import RandomizedPOD
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((1000, 200))  # 대용량 스냅샷
        >>> rpod = RandomizedPOD(n_modes=20)
        >>> rpod.fit(X)
        >>> coeffs = rpod.encode(X)
        >>> X_rec = rpod.decode(coeffs)
    """

    def __init__(
        self,
        n_modes: int = 10,
        n_oversamples: int = 10,
        random_state: int = 42,
        center: bool = True,
    ) -> None:
        """초기화.

        Args:
            n_modes: 보존할 POD 모드 수.
            n_oversamples: 랜덤 오버샘플링 수 (클수록 정밀도 향상, 속도 저하).
            random_state: 난수 시드 (재현성).
            center: True이면 평균 필드를 제거한 뒤 SVD를 수행한다.
        """
        super().__init__()
        self.n_modes = n_modes
        self.n_oversamples = n_oversamples
        self.random_state = random_state
        self.center = center

        self.modes_: NDArray[np.float64] | None = None
        self.singular_values_: NDArray[np.float64] | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.energy_ratio_: NDArray[np.float64] | None = None
        self._used_randomized: bool = False

    # ------------------------------------------------------------------
    # BaseReducer 추상 메서드 구현
    # ------------------------------------------------------------------

    def fit(self, snapshots: NDArray[np.float64]) -> None:
        """스냅샷 행렬로 POD 모드를 계산한다.

        Args:
            snapshots: 스냅샷 행렬. shape = (n_features, n_snapshots).

        Raises:
            ValueError: n_modes가 n_snapshots보다 큰 경우.
        """
        if snapshots.ndim != 2:
            raise ValueError(
                f"snapshots는 2D 배열이어야 합니다. 현재 shape: {snapshots.shape}"
            )

        n_features, n_snapshots = snapshots.shape
        n_modes = min(self.n_modes, n_snapshots, n_features)

        if self.n_modes > n_snapshots:
            logger.warning(
                "n_modes(%d) > n_snapshots(%d) — n_modes를 %d로 줄입니다.",
                self.n_modes,
                n_snapshots,
                n_modes,
            )

        # 평균 제거
        if self.center:
            self.mean_ = snapshots.mean(axis=1)
            X = snapshots - self.mean_[:, np.newaxis]
        else:
            self.mean_ = np.zeros(n_features, dtype=snapshots.dtype)
            X = snapshots

        # SVD 수행 — sklearn randomized_svd 우선, 실패 시 numpy 폴백
        U, s = self._svd(X, n_modes)

        self.modes_ = U[:, :n_modes]
        self.singular_values_ = s[:n_modes]

        # 에너지 누적 기여율 (전체 에너지는 numpy SVD로 별도 계산 불가 → 근사값 사용)
        # randomized_svd는 상위 n_modes 특이값만 반환하므로 해당 범위 내에서 정규화
        total_energy = float(np.sum(s ** 2))
        if total_energy > 0.0:
            cumulative = np.cumsum(s[:n_modes] ** 2) / total_energy
        else:
            cumulative = np.ones(n_modes, dtype=np.float64)
        self.energy_ratio_ = cumulative.astype(np.float64)

        self.n_components = n_modes
        self.is_fitted = True
        logger.info(
            "RandomizedPOD 학습 완료: n_modes=%d, 누적 에너지(근사)=%.4f, "
            "randomized_svd=%s",
            n_modes,
            float(self.energy_ratio_[-1]),
            self._used_randomized,
        )

    def _svd(
        self, X: NDArray[np.float64], n_modes: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """SVD를 수행한다. sklearn 우선, 실패 시 numpy 폴백.

        Args:
            X: 입력 행렬. shape = (n_features, n_snapshots).
            n_modes: 추출할 모드 수.

        Returns:
            (U, s) 튜플. U: (n_features, n_modes), s: (n_modes,).
        """
        try:
            from sklearn.utils.extmath import randomized_svd  # type: ignore[import]

            U, s, _ = randomized_svd(
                X,
                n_components=n_modes,
                n_oversamples=self.n_oversamples,
                random_state=self.random_state,
            )
            self._used_randomized = True
            logger.debug("sklearn randomized_svd 사용.")
            return U, s
        except ImportError:
            logger.warning("sklearn 미설치 — numpy.linalg.svd 로 폴백합니다.")
            U_full, s_full, _ = np.linalg.svd(X, full_matrices=False)
            self._used_randomized = False
            return U_full, s_full

    def encode(self, snapshots: NDArray[np.float64]) -> NDArray[np.float64]:
        """스냅샷을 POD 계수 공간으로 투영한다.

        Args:
            snapshots: 인코딩할 스냅샷. shape = (n_features, n_snapshots).

        Returns:
            POD 계수 행렬. shape = (n_snapshots, n_modes).

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        self._check_fitted()
        assert self.modes_ is not None
        assert self.mean_ is not None

        if snapshots.ndim == 1:
            snapshots = snapshots[:, np.newaxis]

        X = snapshots - self.mean_[:, np.newaxis]
        coeffs = self.modes_.T @ X   # (n_modes, n_snapshots)
        return coeffs.T              # (n_snapshots, n_modes)

    def decode(self, coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        """POD 계수를 원래 차원의 유동장으로 복원한다.

        Args:
            coefficients: POD 계수. shape = (n_snapshots, n_modes).

        Returns:
            복원된 스냅샷. shape = (n_features, n_snapshots).

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        self._check_fitted()
        assert self.modes_ is not None
        assert self.mean_ is not None

        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, :]

        reconstructed = self.modes_ @ coefficients.T  # (n_features, n_snapshots)
        return reconstructed + self.mean_[:, np.newaxis]

    def reconstruct(
        self,
        snapshots: NDArray[np.float64],
        n_modes: int | None = None,
    ) -> NDArray[np.float64]:
        """지정한 모드 수로 스냅샷을 재구성한다.

        Args:
            snapshots: 재구성할 스냅샷. shape = (n_features, n_snapshots).
            n_modes: 사용할 모드 수. None이면 전체 모드 사용.

        Returns:
            재구성된 스냅샷. shape = (n_features, n_snapshots).

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
            ValueError: n_modes > n_components인 경우.
        """
        self._check_fitted()
        assert self.modes_ is not None
        assert self.mean_ is not None

        if n_modes is None:
            n_modes = self.n_components
        if n_modes > self.n_components:
            raise ValueError(
                f"n_modes({n_modes})는 n_components({self.n_components})보다 클 수 없습니다."
            )

        if snapshots.ndim == 1:
            snapshots = snapshots[:, np.newaxis]

        X = snapshots - self.mean_[:, np.newaxis]
        coeffs = self.modes_[:, :n_modes].T @ X
        rec = self.modes_[:, :n_modes] @ coeffs
        return rec + self.mean_[:, np.newaxis]

    @property
    def energy_ratio(self) -> NDArray[np.float64]:
        """각 모드의 누적 에너지 기여율 (근사).

        Returns:
            누적 에너지 비율 배열. shape = (n_modes,).

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        self._check_fitted()
        assert self.energy_ratio_ is not None
        return self.energy_ratio_

    def get_params(self) -> dict:
        """하이퍼파라미터 딕셔너리 반환.

        Returns:
            {"n_modes": ..., "n_oversamples": ..., "random_state": ..., "center": ...}
        """
        return {
            "n_modes": self.n_modes,
            "n_oversamples": self.n_oversamples,
            "random_state": self.random_state,
            "center": self.center,
        }
