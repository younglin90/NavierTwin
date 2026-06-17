"""스파스 센서 + DMD 재구성 파이프라인.

POD 기저로 최적 센서를 배치하고, DMD 로 시간 예측 후
스파스 측정에서 전체 장(field)을 복원하는 end-to-end 워크플로.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 40))  # 50 공간, 40 스냅샷
    >>> from naviertwin.core.sampling.sensor_dmd_pipeline import SensorDMDPipeline
    >>> pipe = SensorDMDPipeline(n_modes=5, n_sensors=8)
    >>> pipe.fit(X)
    >>> rec = pipe.reconstruct_from_sensors(X[:, :1], pipe.sensors)
    >>> rec.shape
    (50, 1)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.linalg.svd_utils import truncated_svd
from naviertwin.core.sampling.sparse_sensor import reconstruct, select_sensors
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class SensorDMDPipeline:
    """POD 기저 기반 스파스 센서 배치 + 장 재구성 파이프라인.

    Attributes:
        n_modes: 사용할 POD 모드 수.
        n_sensors: 배치할 센서 수. n_modes 이상이어야 한다.
        sensors: 선택된 센서 인덱스 (fit 후 설정됨).
        basis: POD 기저 행렬 (n_space, n_modes).
    """

    def __init__(self, n_modes: int = 10, n_sensors: int | None = None) -> None:
        if n_modes <= 0:
            raise ValueError(f"n_modes must be > 0, got {n_modes}")
        self.n_modes = n_modes
        self.n_sensors = n_sensors if n_sensors is not None else n_modes + 2
        self.sensors: NDArray[np.intp] | None = None
        self.basis: NDArray[np.float64] | None = None
        self._singular_values: NDArray[np.float64] | None = None
        self.is_fitted: bool = False

    def fit(self, X: NDArray[np.float64]) -> None:
        """스냅샷 행렬로 POD 기저와 최적 센서 위치를 학습한다.

        Args:
            X: (n_space, n_snapshots) 스냅샷 행렬.

        Raises:
            ValueError: X가 2D가 아닌 경우.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_space, n_snapshots), got {X.shape}")

        n_space, n_snap = X.shape
        r = min(self.n_modes, n_space, n_snap)

        # SVD → POD 모드
        U, s, _ = truncated_svd(X - X.mean(axis=1, keepdims=True), k=r)
        self.basis = U
        self._singular_values = s

        # 센서 수 보정 (n_modes 이상 확보)
        n_sens = max(self.n_sensors, r)
        n_sens = min(n_sens, n_space)
        self.sensors = select_sensors(self.basis, n_sensors=n_sens)
        self.is_fitted = True
        logger.info(
            "SensorDMDPipeline 학습: n_space=%d, n_modes=%d, n_sensors=%d",
            n_space, r, n_sens,
        )

    def reconstruct_from_sensors(
        self,
        field: NDArray[np.float64],
        sensors: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        """센서 위치에서의 측정값으로 전체 장을 재구성한다.

        Args:
            field: (n_space, n_samples) 또는 (n_space,) 전체 장
                   (센서 위치의 값만 사용됨).
            sensors: 센서 인덱스 배열. None이면 fit 시 결정된 센서 사용.

        Returns:
            (n_space, n_samples) 또는 (n_space,) 재구성된 장.

        Raises:
            RuntimeError: fit() 호출 전인 경우.
        """
        if not self.is_fitted or self.basis is None or self.sensors is None:
            raise RuntimeError("fit() must be called before reconstruct_from_sensors()")

        field = np.asarray(field, dtype=np.float64)
        if sensors is None:
            sensors = self.sensors

        squeeze = field.ndim == 1
        if squeeze:
            field = field[:, None]

        y = field[sensors, :]
        rec = reconstruct(self.basis, sensors, y)
        return rec[:, 0] if squeeze else rec

    def energy_fraction(self) -> NDArray[np.float64]:
        """각 POD 모드의 에너지 분율을 반환한다.

        Returns:
            (n_modes,) 에너지 분율 배열 (합계 = 1).

        Raises:
            RuntimeError: fit() 호출 전인 경우.
        """
        if self._singular_values is None:
            raise RuntimeError("fit() must be called first")
        s2 = self._singular_values ** 2
        return s2 / s2.sum()


__all__ = ["SensorDMDPipeline"]
