"""Snapshot POD (Proper Orthogonal Decomposition).

스냅샷 행렬에 SVD를 적용하여 에너지 최적 POD 모드를 추출한다.
NumPy SVD를 기본으로 사용하며, sklearn randomized_svd를 선택적으로 활용한다.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.dimensionality_reduction.base import BaseReducer
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class SnapshotPOD(BaseReducer):
    """스냅샷 POD (SVD 기반 차원 축소).

    스냅샷 행렬 X = (n_features, n_snapshots) 에 대해
    SVD를 수행하여 에너지 최적 모드를 추출한다.

    Attributes:
        n_modes: 보존할 모드 수.
        center: True이면 평균 필드를 제거한 뒤 SVD를 수행한다.
        modes_: shape (n_features, n_modes) POD 모드 (fit 후 설정).
        singular_values_: shape (n_modes,) 특이값 (fit 후 설정).
        mean_: shape (n_features,) 평균 필드 (fit 후 설정).
        energy_ratio_: shape (n_modes,) 누적 에너지 기여율 (fit 후 설정).

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((100, 50))  # (n_features, n_snapshots)
        >>> pod = SnapshotPOD(n_modes=5)
        >>> pod.fit(X)
        >>> coeffs = pod.encode(X)   # (n_snapshots, n_modes) or (n_features, n_modes)
        >>> X_rec = pod.decode(coeffs)
    """

    def __init__(self, n_modes: int = 10, center: bool = True) -> None:
        """초기화.

        Args:
            n_modes: 보존할 POD 모드 수.
            center: True이면 평균 필드를 제거한 뒤 SVD를 수행한다.
        """
        super().__init__()
        self.n_modes = n_modes
        self.center = center

        # fit 후 설정되는 속성
        self.modes_: NDArray[np.float64] | None = None
        self.singular_values_: NDArray[np.float64] | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.energy_ratio_: NDArray[np.float64] | None = None

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
            self.mean_ = snapshots.mean(axis=1)  # (n_features,)
            X = snapshots - self.mean_[:, np.newaxis]
        else:
            self.mean_ = np.zeros(n_features, dtype=snapshots.dtype)
            X = snapshots

        logger.debug(
            "SnapshotPOD.fit: n_features=%d, n_snapshots=%d, n_modes=%d",
            n_features,
            n_snapshots,
            n_modes,
        )

        # SVD 수행
        U, s, _ = np.linalg.svd(X, full_matrices=False)

        # 상위 n_modes 모드 보존
        self.modes_ = U[:, :n_modes]                   # (n_features, n_modes)
        self.singular_values_ = s[:n_modes]            # (n_modes,)

        # 에너지 누적 기여율 계산 (전체 특이값 기준)
        total_energy = float(np.sum(s**2))
        if total_energy > 0.0:
            cumulative = np.cumsum(s[:n_modes] ** 2) / total_energy
        else:
            cumulative = np.ones(n_modes, dtype=np.float64)
        self.energy_ratio_ = cumulative.astype(np.float64)

        self.n_components = n_modes
        self.is_fitted = True
        logger.info(
            "SnapshotPOD 학습 완료: n_modes=%d, 누적 에너지=%.4f",
            n_modes,
            float(self.energy_ratio_[-1]),
        )

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

        X = snapshots - self.mean_[:, np.newaxis]  # 평균 제거
        coeffs = self.modes_.T @ X                  # (n_modes, n_snapshots)
        return coeffs.T                             # (n_snapshots, n_modes)

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

        # (n_features, n_modes) @ (n_modes, n_snapshots) → (n_features, n_snapshots)
        reconstructed = self.modes_ @ coefficients.T
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
        coeffs = self.modes_[:, :n_modes].T @ X    # (n_modes, n_snapshots)
        rec = self.modes_[:, :n_modes] @ coeffs    # (n_features, n_snapshots)
        return rec + self.mean_[:, np.newaxis]

    @property
    def energy_ratio(self) -> NDArray[np.float64]:
        """각 모드의 누적 에너지 기여율.

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
            {"n_modes": ..., "center": ...}
        """
        return {"n_modes": self.n_modes, "center": self.center}
