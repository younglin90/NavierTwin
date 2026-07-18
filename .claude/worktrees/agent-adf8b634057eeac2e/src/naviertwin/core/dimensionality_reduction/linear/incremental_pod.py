"""점증적 POD (Incremental / Streaming POD).

새 스냅샷이 도착할 때마다 기저를 갱신하는 온라인 POD.
Brand (2006)의 SVD 갱신 알고리즘 기반 경량 구현.

References:
    Brand, M., "Fast low-rank modifications of the thin singular value
    decomposition", Linear Algebra and Its Applications, 2006.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD
    >>> pod = IncrementalPOD(n_modes=5)
    >>> step = 0
    >>> while step < 10:
    ...     snap = rng.standard_normal(50)
    ...     pod.update(snap)
    ...     step += 1
    >>> pod.basis.shape
    (50, 5)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class IncrementalPOD:
    """온라인 점증적 POD — 스냅샷 스트림에서 기저를 갱신한다.

    Attributes:
        n_modes: 유지할 최대 POD 모드 수.
        basis: (n_space, n_modes) POD 기저 (직교 정규).
        singular_values: (n_modes,) 특이값.
        n_snapshots: 처리된 스냅샷 수.
    """

    def __init__(self, n_modes: int = 10, forget_factor: float = 1.0) -> None:
        """초기화.

        Args:
            n_modes: 유지할 POD 모드 수.
            forget_factor: 이전 데이터 감쇠 인수 (0 < λ ≤ 1). 1.0이면 망각 없음.

        Raises:
            ValueError: n_modes ≤ 0 또는 forget_factor 범위 오류.
        """
        if n_modes <= 0:
            raise ValueError(f"n_modes must be > 0, got {n_modes}")
        if not (0 < forget_factor <= 1.0):
            raise ValueError(
                f"forget_factor must be in (0, 1], got {forget_factor}"
            )
        self.n_modes = n_modes
        self.forget_factor = forget_factor
        self.basis: NDArray[np.float64] | None = None
        self.singular_values: NDArray[np.float64] | None = None
        self.n_snapshots: int = 0
        self._mean: NDArray[np.float64] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.basis is not None

    @property
    def n_components(self) -> int:
        """현재 유지 중인 모드 수."""
        return int(self.basis.shape[1]) if self.basis is not None else 0

    def fit(self, snapshots: NDArray[np.float64]) -> "IncrementalPOD":
        """오프라인 스냅샷 배치로 초기 학습한다."""
        self.basis = None
        self.singular_values = None
        self.n_snapshots = 0
        self._mean = None
        self.batch_update(snapshots)
        return self

    def update(self, snapshot: NDArray[np.float64]) -> None:
        """새 스냅샷으로 POD 기저를 갱신한다.

        Args:
            snapshot: (n_space,) 새 스냅샷 벡터.

        Raises:
            ValueError: snapshot이 1D가 아닌 경우 또는 공간 차원 불일치.
        """
        x = np.asarray(snapshot, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"snapshot must be 1D, got shape {x.shape}")

        n = self.n_snapshots

        # 평균 업데이트 (운닝 평균)
        if self._mean is None:
            self._mean = x.copy()
        else:
            if x.shape != self._mean.shape:
                raise ValueError(
                    f"snapshot shape {x.shape} != existing shape {self._mean.shape}"
                )
            self._mean = (n * self._mean + x) / (n + 1)

        self.n_snapshots += 1
        xc = x - self._mean  # 중심화

        if self.basis is None:
            # 첫 번째 스냅샷: 직접 초기화
            norm = np.linalg.norm(xc)
            if norm > 1e-14:
                self.basis = (xc / norm)[:, None]
                self.singular_values = np.array([norm])
            else:
                self.basis = np.zeros((len(x), 1))
                self.singular_values = np.array([0.0])
            self._refresh_compat_attrs()
            return

        # 기존 기저에 투영
        U = self.basis
        s = self.singular_values * self.forget_factor

        # Brand's rank-1 update
        m = U.T @ xc  # 투영 계수
        p = xc - U @ m  # 잔차
        Ra = np.diag(s)
        norm_p = np.linalg.norm(p)

        if norm_p > 1e-14:
            p_hat = p / norm_p
            # 확장 행렬 K
            K = np.zeros((Ra.shape[0] + 1, Ra.shape[1] + 1))
            K[: Ra.shape[0], : Ra.shape[1]] = Ra
            K[: Ra.shape[0], -1] = m
            K[-1, -1] = norm_p
        else:
            p_hat = None
            K = np.zeros((Ra.shape[0], Ra.shape[1] + 1))
            K[:, : Ra.shape[1]] = Ra
            K[:, -1] = m

        # K의 SVD
        Uk, sk, _ = _svd(K, full_matrices=False)

        # 기저 갱신
        if p_hat is not None:
            U_new = np.hstack([U, p_hat[:, None]]) @ Uk
        else:
            U_new = U @ Uk

        # n_modes 유지
        r = min(self.n_modes, U_new.shape[1])
        self.basis = U_new[:, :r]
        self.singular_values = sk[:r]
        self._refresh_compat_attrs()

    def batch_update(self, snapshots: NDArray[np.float64]) -> None:
        """여러 스냅샷 배치를 순차 업데이트한다.

        Args:
            snapshots: (n_space, n_snapshots) 또는 (n_snapshots, n_space) 배열.
                       n_space > n_snapshots이면 (n_space, n_snapshots)로 해석.

        Raises:
            ValueError: snapshots가 2D가 아닌 경우.
        """
        S = np.asarray(snapshots, dtype=np.float64)
        if S.ndim != 2:
            raise ValueError(f"snapshots must be 2D, got {S.shape}")

        # 첫 업데이트 시 공간 차원 추정
        if S.shape[0] > S.shape[1]:
            # (n_space, n_snap) 형식
            j = 0
            while j < S.shape[1]:
                self.update(S[:, j])
                j += 1
        else:
            # (n_snap, n_space) 형식
            j = 0
            while j < S.shape[0]:
                self.update(S[j, :])
                j += 1

    def project(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        """필드를 현재 POD 기저에 투영해 계수를 반환한다.

        Args:
            field: (n_space,) 또는 (n_space, n_samples) 필드.

        Returns:
            (n_modes,) 또는 (n_modes, n_samples) 투영 계수.

        Raises:
            RuntimeError: update() 미호출.
        """
        if self.basis is None:
            raise RuntimeError("No snapshots processed yet; call update() first")
        field = np.asarray(field, dtype=np.float64)
        mean = self._mean if self._mean is not None else 0.0
        if field.ndim == 1:
            return self.basis.T @ (field - mean)
        return self.basis.T @ (
            field - (self._mean[:, None] if self._mean is not None else 0.0)
        )

    def encode(self, snapshots: NDArray[np.float64]) -> NDArray[np.float64]:
        """스냅샷을 잠재 계수로 인코딩한다.

        Returns:
            (n_samples, n_modes) 계수 행렬.
        """
        if self.basis is None:
            raise RuntimeError("No snapshots processed yet; call update() first")
        X = np.asarray(snapshots, dtype=np.float64)
        n_space = self.basis.shape[0]
        if X.ndim == 1:
            return self.project(X)[None, :]
        if X.ndim != 2:
            raise ValueError(f"snapshots must be 1D/2D, got {X.shape}")

        if X.shape[0] == n_space:
            return self.project(X).T
        if X.shape[1] == n_space:
            return self.project(X.T).T
        raise ValueError(f"snapshot shape {X.shape} incompatible with n_space={n_space}")

    def decode(self, coeffs: NDArray[np.float64]) -> NDArray[np.float64]:
        """잠재 계수에서 원공간 필드를 복원한다."""
        if self.basis is None:
            raise RuntimeError("No snapshots processed yet; call update() first")
        C = np.asarray(coeffs, dtype=np.float64)
        mean = self._mean if self._mean is not None else np.zeros(self.basis.shape[0])
        n_modes = self.basis.shape[1]

        if C.ndim == 1:
            if C.shape[0] != n_modes:
                raise ValueError(f"coeff shape {C.shape} incompatible with n_modes={n_modes}")
            return self.basis @ C + mean
        if C.ndim != 2:
            raise ValueError(f"coeffs must be 1D/2D, got {C.shape}")

        if C.shape[1] == n_modes:
            return self.basis @ C.T + mean[:, None]
        if C.shape[0] == n_modes:
            return self.basis @ C + mean[:, None]
        raise ValueError(f"coeff shape {C.shape} incompatible with n_modes={n_modes}")

    def energy_fraction(self) -> NDArray[np.float64]:
        """각 모드의 에너지 분율을 반환한다.

        Returns:
            (n_modes,) 에너지 분율 배열.

        Raises:
            RuntimeError: update() 미호출.
        """
        if self.singular_values is None:
            raise RuntimeError("No snapshots processed yet; call update() first")
        s2 = self.singular_values ** 2
        total = s2.sum()
        if total < 1e-30:
            return np.zeros_like(s2)
        return s2 / total

    @property
    def energy_ratio(self) -> NDArray[np.float64]:
        """BaseReducer 호환 에너지 비율."""
        return self.energy_fraction()

    def _refresh_compat_attrs(self) -> None:
        """기존 POD 계열과의 속성 호환 필드를 갱신한다."""
        if self.basis is None or self.singular_values is None:
            return
        self.modes_ = self.basis
        self.singular_values_ = self.singular_values
        self.mean_ = self._mean if self._mean is not None else np.zeros(self.basis.shape[0])
        self.energy_ratio_ = self.energy_fraction()


__all__ = ["IncrementalPOD"]
