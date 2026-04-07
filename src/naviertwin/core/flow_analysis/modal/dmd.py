"""Dynamic Mode Decomposition (PyDMD 래퍼).

PyDMD 라이브러리를 이용하여 DMD 분석을 수행한다.
pydmd 미설치 시 ImportError를 발생시키며, 사용 전에
``pytest.importorskip("pydmd")`` 또는 try/except 처리를 권장한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

# 지원하는 PyDMD 클래스 이름 매핑
_PYDMD_CLASSES: dict[str, str] = {
    "dmd": "DMD",
    "fbdmd": "FbDMD",
    "hodmd": "HODMD",
    "spdmd": "SpDMD",
}


class DMDAnalyzer:
    """PyDMD 기반 DMD 분석기.

    스냅샷 행렬 X = (n_features, n_snapshots) 에서
    동적 모드, 고유값(주파수/성장률), 진폭을 추출한다.

    Attributes:
        method: DMD 변형. "dmd" | "fbdmd" | "hodmd" | "spdmd" (기본: "fbdmd").
        dt: 타임스텝 간격 [s].
        n_modes: 추출할 모드 수. None이면 전체 모드 사용.

    Examples:
        >>> import numpy as np
        >>> from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((50, 100))
        >>> analyzer = DMDAnalyzer(method="dmd", dt=0.01)
        >>> analyzer.fit(X)
        >>> freqs = analyzer.frequencies
        >>> df = analyzer.get_spectrum_dataframe()
    """

    def __init__(
        self,
        method: str = "fbdmd",
        dt: float = 1.0,
        n_modes: int | None = None,
    ) -> None:
        """초기화.

        Args:
            method: PyDMD 변형 종류.
                "dmd" | "fbdmd" | "hodmd" | "spdmd".
            dt: 타임스텝 간격 [s]. 주파수 계산에 사용.
            n_modes: 추출할 모드 수. None이면 전체 모드 사용.

        Raises:
            ValueError: method가 지원되지 않는 경우.
        """
        if method not in _PYDMD_CLASSES:
            raise ValueError(
                f"지원되지 않는 method: '{method}'. "
                f"지원 목록: {list(_PYDMD_CLASSES.keys())}"
            )
        self.method = method
        self.dt = dt
        self.n_modes = n_modes
        self._dmd: Any = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def fit(self, snapshots: NDArray[np.float64]) -> None:
        """PyDMD로 DMD를 수행한다.

        Args:
            snapshots: 스냅샷 행렬. shape = (n_features, n_snapshots).

        Raises:
            ImportError: pydmd 미설치 시.
            ValueError: snapshots가 2D가 아닌 경우.
        """
        try:
            import pydmd  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pydmd 패키지가 설치되어 있지 않습니다. "
                "`pip install pydmd` 로 설치하세요."
            ) from exc

        if snapshots.ndim != 2:
            raise ValueError(
                f"snapshots는 2D 배열이어야 합니다. 현재 shape: {snapshots.shape}"
            )

        n_features, n_snapshots = snapshots.shape
        logger.debug(
            "DMDAnalyzer.fit: method=%s, n_features=%d, n_snapshots=%d",
            self.method,
            n_features,
            n_snapshots,
        )

        cls_name = _PYDMD_CLASSES[self.method]
        dmd_cls = getattr(pydmd, cls_name)

        # svd_rank=0 이면 PyDMD가 자동으로 랭크 결정
        svd_rank = self.n_modes if self.n_modes is not None else 0

        # HODMD는 d(지연 시간) 파라미터가 필요
        if self.method == "hodmd":
            self._dmd = dmd_cls(svd_rank=svd_rank, d=10)
        else:
            self._dmd = dmd_cls(svd_rank=svd_rank)

        self._dmd.fit(snapshots)
        self._is_fitted = True

        logger.info(
            "DMDAnalyzer 학습 완료: method=%s, n_modes=%d",
            self.method,
            self.modes.shape[1],
        )

    # ------------------------------------------------------------------
    # 프로퍼티
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """fit() 완료 여부를 확인한다.

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        if not self._is_fitted or self._dmd is None:
            raise RuntimeError("predict() 전에 fit()을 먼저 호출해야 합니다.")

    @property
    def modes(self) -> NDArray[np.complex128]:
        """DMD 모드.

        Returns:
            shape (n_features, n_modes) 복소 모드 행렬.

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        self._check_fitted()
        return self._dmd.modes  # (n_features, n_modes)

    @property
    def eigenvalues(self) -> NDArray[np.complex128]:
        """이산 고유값 (단위원 내외 위치로 안정성 판단).

        Returns:
            shape (n_modes,) 복소 고유값 배열.

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        self._check_fitted()
        return self._dmd.eigs  # (n_modes,)

    @property
    def frequencies(self) -> NDArray[np.float64]:
        """연속 주파수 [Hz].

        이산 고유값의 허수부로부터 연속 주파수를 계산한다:
        f = Im(log(λ)) / (2π * dt)

        Returns:
            shape (n_modes,) 주파수 배열 [Hz].

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        self._check_fitted()
        eigs = self.eigenvalues
        omega = np.log(eigs) / self.dt   # 연속 고유값
        return (np.imag(omega) / (2.0 * np.pi)).astype(np.float64)

    @property
    def growth_rates(self) -> NDArray[np.float64]:
        """성장률 (양수: 불안정, 음수: 감쇠).

        이산 고유값의 절대값 로그로 계산한다:
        σ = Re(log(λ)) / dt

        Returns:
            shape (n_modes,) 성장률 배열.

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        self._check_fitted()
        eigs = self.eigenvalues
        omega = np.log(eigs) / self.dt
        return np.real(omega).astype(np.float64)

    @property
    def amplitudes(self) -> NDArray[np.float64]:
        """모드 진폭 (초기 조건에서 계산된 계수 절대값).

        Returns:
            shape (n_modes,) 진폭 배열.

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        self._check_fitted()
        return np.abs(self._dmd.amplitudes).astype(np.float64)

    # ------------------------------------------------------------------
    # 재구성 및 분석
    # ------------------------------------------------------------------

    def reconstruct(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """시간 배열 t에서 유동장을 재구성한다.

        Args:
            t: 재구성할 시간 배열. shape = (n_times,) [s].

        Returns:
            재구성된 유동장. shape = (n_features, n_times) 실수 부분.

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        self._check_fitted()

        eigs = self.eigenvalues         # (n_modes,)
        amps = self._dmd.amplitudes     # (n_modes,)
        modes = self.modes              # (n_features, n_modes)

        # 각 시간 t에서 재구성: Σ_r amplitude_r * mode_r * λ_r^(t/dt)
        n_features = modes.shape[0]
        n_times = len(t)
        field = np.zeros((n_features, n_times), dtype=np.complex128)

        for i, ti in enumerate(t):
            exponents = eigs ** (ti / self.dt)  # (n_modes,)
            field[:, i] = modes @ (amps * exponents)

        return np.real(field).astype(np.float64)

    def get_spectrum_dataframe(self) -> Any:
        """주파수, 성장률, 진폭을 담은 pandas DataFrame을 반환한다.

        Returns:
            pandas.DataFrame with columns:
                - frequency: 연속 주파수 [Hz]
                - growth_rate: 성장률
                - amplitude: 모드 진폭
                - eigenvalue_real: 고유값 실수부
                - eigenvalue_imag: 고유값 허수부

        Raises:
            ImportError: pandas 미설치 시.
            RuntimeError: fit()이 호출되지 않은 경우.
        """
        try:
            import pandas as pd  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pandas 패키지가 설치되어 있지 않습니다. "
                "`pip install pandas` 로 설치하세요."
            ) from exc

        self._check_fitted()
        eigs = self.eigenvalues

        return pd.DataFrame(
            {
                "frequency": self.frequencies,
                "growth_rate": self.growth_rates,
                "amplitude": self.amplitudes,
                "eigenvalue_real": np.real(eigs),
                "eigenvalue_imag": np.imag(eigs),
            }
        )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"DMDAnalyzer(method='{self.method}', dt={self.dt}, "
            f"n_modes={self.n_modes}, status={status})"
        )
