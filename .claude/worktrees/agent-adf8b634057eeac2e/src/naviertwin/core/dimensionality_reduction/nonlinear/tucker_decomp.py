"""Tucker 텐서 분해 기반 3D 필드 차원축소.

3D 유동장 X[t, i, j] → core G[r_t, r_i, r_j] + 세 factor 행렬.
HOSVD 초기화 + HOOI 반복 최적화.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.tucker_decomp import (
    ...     TuckerDecomposition,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> T = rng.standard_normal((20, 16, 16))
    >>> tk = TuckerDecomposition(ranks=(5, 4, 4))
    >>> tk.fit(T)
    >>> T_rec = tk.reconstruct()
    >>> float(np.linalg.norm(T - T_rec) / np.linalg.norm(T)) < 1.0
    True
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _unfold(T: NDArray[np.float64], mode: int) -> NDArray[np.float64]:
    """텐서를 주어진 mode 축을 0번으로 이동해 (n, -1) 로 펼친다."""
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def _fold(mat: NDArray[np.float64], mode: int, shape: tuple[int, ...]) -> NDArray[np.float64]:
    """_unfold 의 역."""
    full_shape = [shape[mode]] + list(np.delete(np.asarray(shape, dtype=int), mode))
    return np.moveaxis(mat.reshape(full_shape), 0, mode)


def _mode_product(
    T: NDArray[np.float64], M: NDArray[np.float64], mode: int
) -> NDArray[np.float64]:
    """T ×_n M — mode-n 곱."""
    unfolded = _unfold(T, mode)
    new_mat = M @ unfolded
    new_shape = list(T.shape)
    new_shape[mode] = M.shape[0]
    return _fold(new_mat, mode, tuple(new_shape))


class TuckerDecomposition:
    """HOSVD 초기화 + (선택) HOOI 반복 최적화 Tucker 분해."""

    def __init__(
        self,
        ranks: tuple[int, ...],
        max_iter: int = 10,
        tol: float = 1e-6,
    ) -> None:
        self.ranks = tuple(ranks)
        self.max_iter = max_iter
        self.tol = tol

        self.core_: NDArray[np.float64] | None = None
        self.factors_: list[NDArray[np.float64]] = []
        self.is_fitted: bool = False
        self.errors_: list[float] = []

    def fit(self, X: NDArray[np.float64]) -> None:
        """HOSVD → HOOI 로 Tucker 분해를 학습한다."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != len(self.ranks):
            raise ValueError(
                f"X.ndim({X.ndim}) != len(ranks)({len(self.ranks)})"
            )

        # HOSVD 초기 factors
        factors: list[NDArray[np.float64]] = []
        mode = 0
        while mode < X.ndim:
            U, _, _ = _svd(_unfold(X, mode), full_matrices=False)
            factors.append(U[:, : self.ranks[mode]])
            mode += 1

        # HOOI
        prev_err = np.inf
        self.errors_ = []
        iteration = 0
        while iteration < self.max_iter:
            mode = 0
            while mode < X.ndim:
                Y = X
                n = 0
                while n < len(factors):
                    F = factors[n]
                    if n == mode:
                        n += 1
                        continue
                    Y = _mode_product(Y, F.T, n)
                    n += 1
                U, _, _ = _svd(_unfold(Y, mode), full_matrices=False)
                factors[mode] = U[:, : self.ranks[mode]]
                mode += 1

            # core 계산
            core = X
            n = 0
            while n < len(factors):
                F = factors[n]
                core = _mode_product(core, F.T, n)
                n += 1

            # 재구성 오차
            T_rec = core
            n = 0
            while n < len(factors):
                F = factors[n]
                T_rec = _mode_product(T_rec, F, n)
                n += 1
            err = float(np.linalg.norm(X - T_rec))
            self.errors_.append(err)
            if abs(prev_err - err) / max(prev_err, 1e-30) < self.tol:
                break
            prev_err = err
            iteration += 1

        self.factors_ = factors
        self.core_ = core
        self.is_fitted = True
        logger.info(
            "Tucker 분해 완료: ranks=%s, final_err=%.4g",
            self.ranks,
            self.errors_[-1] if self.errors_ else float("nan"),
        )

    def reconstruct(self) -> NDArray[np.float64]:
        if not self.is_fitted or self.core_ is None:
            raise RuntimeError("fit() 먼저 호출하세요")
        T = self.core_
        n = 0
        while n < len(self.factors_):
            F = self.factors_[n]
            T = _mode_product(T, F, n)
            n += 1
        return T

    def compression_ratio(self, original_shape: tuple[int, ...]) -> float:
        """원본 대비 압축률 (원본 / tucker 파라미터)."""
        if not self.is_fitted:
            return 0.0
        n_orig = int(np.prod(original_shape))
        n_core = int(np.prod(self.ranks))
        n_factor = sum(map(lambda pair: pair[0] * pair[1], zip(original_shape, self.ranks)))
        return n_orig / (n_core + n_factor)


__all__ = ["TuckerDecomposition"]
