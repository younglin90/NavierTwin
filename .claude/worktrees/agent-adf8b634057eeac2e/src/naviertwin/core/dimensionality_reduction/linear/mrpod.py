"""다중 해상도 POD (Multi-Resolution POD, MRPOD).

공간 데이터를 해상도별로 분해해 각 스케일의 지배 모드를 추출한다.
블록 SVD 기반으로 구현되며 PyWavelets가 없어도 작동한다.

References:
    Mendez et al., "Multi-Scale Proper Orthogonal Decomposition of Complex
    Fluid Flows", JFM, 2019.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 50))
    >>> from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD
    >>> m = MRPOD(n_scales=3, n_modes_per_scale=4)
    >>> m.fit(X)
    >>> modes = m.get_modes()
    >>> modes.shape[0] == 100
    True
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _low_pass_filter(X: NDArray[np.float64], level: int) -> NDArray[np.float64]:
    """간단한 가우시안 저주파 필터 (스케일별 분리용)."""
    sigma = 2 ** level
    from math import ceil

    kernel_size = int(ceil(6 * sigma)) | 1  # odd
    half = kernel_size // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()

    def _filter_column(col: NDArray[np.float64]) -> NDArray[np.float64]:
        filtered = np.convolve(col, kernel, mode="same")
        if filtered.shape[0] != X.shape[0]:
            start = max(0, (filtered.shape[0] - X.shape[0]) // 2)
            filtered = filtered[start : start + X.shape[0]]
        return filtered

    return np.apply_along_axis(_filter_column, 0, X)


class MRPOD:
    """다중 해상도 POD 분해기.

    Attributes:
        n_scales: 해상도 레벨 수.
        n_modes_per_scale: 각 스케일당 추출할 POD 모드 수.
        scale_modes: 학습 후 각 스케일별 POD 모드 리스트.
        scale_energies: 각 스케일별 특이값².
    """

    def __init__(self, n_scales: int = 3, n_modes_per_scale: int = 5) -> None:
        if n_scales <= 0:
            raise ValueError(f"n_scales must be > 0, got {n_scales}")
        if n_modes_per_scale <= 0:
            raise ValueError(f"n_modes_per_scale must be > 0, got {n_modes_per_scale}")
        self.n_scales = n_scales
        self.n_modes_per_scale = n_modes_per_scale
        self.scale_modes: list[NDArray[np.float64]] = []
        self.scale_energies: list[NDArray[np.float64]] = []
        self.mean_: NDArray[np.float64] | None = None
        self.modes_: NDArray[np.float64] | None = None
        self.singular_values_: NDArray[np.float64] | None = None
        self.energy_ratio_: NDArray[np.float64] | None = None
        self.n_components: int = 0
        self.is_fitted: bool = False

    def fit(self, X: NDArray[np.float64]) -> None:
        """스냅샷 행렬에서 다중 해상도 POD를 수행한다.

        Args:
            X: (n_space, n_snapshots) 스냅샷 행렬.

        Raises:
            ValueError: X가 2D가 아닌 경우.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_space, n_snapshots), got {X.shape}")

        n_space, n_snap = X.shape
        r = min(self.n_modes_per_scale, n_space, n_snap)

        self.scale_modes = []
        self.scale_energies = []

        self.mean_ = X.mean(axis=1, keepdims=True)
        residual = X - self.mean_

        scale = 0
        while scale < self.n_scales:
            # 이 스케일의 저주파 컴포넌트
            low = _low_pass_filter(residual, level=scale)

            # POD: SVD
            U, s, _ = _svd(low, full_matrices=False)
            modes = U[:, :r]
            energies = s[:r] ** 2

            self.scale_modes.append(modes)
            self.scale_energies.append(energies)

            # 잔차 업데이트 (고주파 = 원본 - 저주파)
            residual = residual - low

            logger.debug(
                "MRPOD 스케일 %d: 에너지=%.4g", scale, float(energies.sum())
            )
            scale += 1

        self.is_fitted = True
        self.modes_ = self.get_modes()
        singular_values = list(
            map(lambda energies: np.sqrt(np.maximum(energies, 0.0)), self.scale_energies)
        )
        self.singular_values_ = np.hstack(singular_values) if singular_values else np.array([])
        total = float(np.sum(self.singular_values_ ** 2))
        if total > 0:
            self.energy_ratio_ = (self.singular_values_ ** 2) / total
        else:
            self.energy_ratio_ = np.zeros_like(self.singular_values_)
        self.n_components = int(self.modes_.shape[1])
        logger.info(
            "MRPOD 학습 완료: %d 스케일, %d 모드/스케일", self.n_scales, r
        )

    def get_modes(self) -> NDArray[np.float64]:
        """모든 스케일의 POD 모드를 열-방향으로 결합해 반환한다.

        Returns:
            (n_space, n_scales * n_modes_per_scale) 모드 행렬.

        Raises:
            RuntimeError: fit() 호출 전인 경우.
        """
        if not self.is_fitted:
            raise RuntimeError("fit() must be called first")
        return np.hstack(self.scale_modes)

    def get_energy_fraction(self) -> NDArray[np.float64]:
        """스케일별 에너지 분율 배열을 반환한다.

        Returns:
            (n_scales,) 배열 — 각 스케일의 총 에너지 / 전체 에너지.

        Raises:
            RuntimeError: fit() 호출 전인 경우.
        """
        if not self.is_fitted:
            raise RuntimeError("fit() must be called first")
        per_scale = np.fromiter(map(np.sum, self.scale_energies), dtype=np.float64)
        total = per_scale.sum()
        if total < 1e-30:
            return np.zeros(self.n_scales)
        return per_scale / total

    def reconstruct(
        self, coefficients: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """계수 벡터로 원래 공간을 재구성한다.

        Args:
            coefficients: (n_total_modes,) 계수 배열
                          n_total_modes = n_scales × n_modes_per_scale.

        Returns:
            (n_space,) 재구성된 필드.

        Raises:
            RuntimeError: fit() 호출 전인 경우.
            ValueError: 계수 길이가 맞지 않는 경우.
        """
        modes = self.get_modes()  # raises if not fitted
        if coefficients.shape != (modes.shape[1],):
            raise ValueError(
                f"coefficients shape {coefficients.shape} != ({modes.shape[1]},)"
            )
        mean = self.mean_.ravel() if self.mean_ is not None else 0.0
        return modes @ coefficients + mean

    def encode(self, snapshots: NDArray[np.float64]) -> NDArray[np.float64]:
        """스냅샷을 잠재 계수로 인코딩한다.

        Returns:
            (n_samples, n_modes) 계수 행렬.
        """
        if not self.is_fitted:
            raise RuntimeError("fit() must be called first")
        X = np.asarray(snapshots, dtype=np.float64)
        modes = self.get_modes()
        mean = self.mean_ if self.mean_ is not None else np.zeros((modes.shape[0], 1))

        if X.ndim == 1:
            coeff = modes.T @ (X - mean.ravel())
            return coeff[None, :]
        if X.ndim != 2:
            raise ValueError(f"snapshots must be 1D/2D, got {X.shape}")

        if X.shape[0] == modes.shape[0]:
            coeffs = modes.T @ (X - mean)
            return coeffs.T
        if X.shape[1] == modes.shape[0]:
            return (X - mean.ravel()[None, :]) @ modes
        raise ValueError(
            f"snapshot shape {X.shape} incompatible with n_space={modes.shape[0]}"
        )

    def decode(self, coeffs: NDArray[np.float64]) -> NDArray[np.float64]:
        """잠재 계수에서 원공간 필드를 복원한다."""
        if not self.is_fitted:
            raise RuntimeError("fit() must be called first")
        C = np.asarray(coeffs, dtype=np.float64)
        modes = self.get_modes()
        mean = self.mean_ if self.mean_ is not None else np.zeros((modes.shape[0], 1))

        if C.ndim == 1:
            if C.shape[0] != modes.shape[1]:
                raise ValueError(
                    f"coefficients shape {C.shape} != ({modes.shape[1]},)"
                )
            return modes @ C + mean.ravel()
        if C.ndim != 2:
            raise ValueError(f"coeffs must be 1D/2D, got {C.shape}")

        if C.shape[1] == modes.shape[1]:
            return modes @ C.T + mean
        if C.shape[0] == modes.shape[1]:
            return modes @ C + mean
        raise ValueError(
            f"coefficients shape {C.shape} incompatible with n_modes={modes.shape[1]}"
        )


__all__ = ["MRPOD"]
