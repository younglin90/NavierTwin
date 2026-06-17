"""PyKoopman / pykoop 래퍼 — 있으면 그걸 쓰고 없으면 자체 DMD.

사용자 친화적 Koopman 분석: fit(time_series) → K 행렬 + 고유값.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_analysis.modal.pykoopman_wrapper import KoopmanAnalysis
    >>> rng = np.random.default_rng(0)
    >>> # Linear system: x_{t+1} = A x_t
    >>> A = np.array([[0.95, 0.02], [-0.01, 0.9]])
    >>> traj = [rng.standard_normal(2)]
    >>> step = 0
    >>> while step < 50:
    ...     traj.append(A @ traj[-1])
    ...     step += 1
    >>> X = np.array(traj)
    >>> ka = KoopmanAnalysis()
    >>> ka.fit(X)
    >>> eigs = ka.eigenvalues()
    >>> eigs.shape
    (2,)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class KoopmanAnalysis:
    """PyKoopman 선호 + numpy DMD 폴백."""

    def __init__(self, rank: int | None = None) -> None:
        self.rank = rank
        self.K_: NDArray[np.float64] | None = None
        self._use_pykoopman: bool = False
        self._pykoopman_model: object | None = None

    def fit(self, X: NDArray[np.float64]) -> None:
        """X: (T, d) 시계열."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X (T, d) T>=2 필요")

        try:
            import pykoopman as pk

            model = pk.Koopman()
            model.fit(X)
            self._pykoopman_model = model
            self._use_pykoopman = True
            self.K_ = np.asarray(model.A, dtype=np.float64)
            logger.info("KoopmanAnalysis(pykoopman): K=%s", self.K_.shape)
            return
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("pykoopman 실패 → DMD: %s", e)

        # DMD fallback (Exact DMD)
        X0 = X[:-1].T
        X1 = X[1:].T
        r = self.rank or min(X0.shape[0], X0.shape[1])
        U, s, Vt = _svd(X0, full_matrices=False)
        U_r = U[:, :r]
        S_r = np.diag(s[:r])
        V_r = Vt[:r].T
        A_tilde = U_r.T @ X1 @ V_r @ np.linalg.pinv(S_r)
        self.K_ = U_r @ A_tilde @ U_r.T
        logger.info("KoopmanAnalysis(DMD): K=%s, rank=%d", self.K_.shape, r)

    def eigenvalues(self) -> NDArray[np.complex128]:
        if self.K_ is None:
            raise RuntimeError("fit() 먼저 호출")
        return np.linalg.eigvals(self.K_)

    def predict(self, x0: NDArray[np.float64], n_steps: int) -> NDArray[np.float64]:
        if self.K_ is None:
            raise RuntimeError("fit() 먼저 호출")
        x = np.asarray(x0, dtype=np.float64).ravel()
        out = []
        step = 0
        while step < n_steps:
            x = self.K_ @ x
            out.append(x.copy())
            step += 1
        return np.stack(out)


__all__ = ["KoopmanAnalysis"]
