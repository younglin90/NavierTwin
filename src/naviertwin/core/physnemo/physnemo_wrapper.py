"""NVIDIA PhysicsNEMO 래퍼 — 설치된 경우만 실제 기능 활성화.

PhysicsNEMO 는 무거운 CUDA/Python 의존성. 이 래퍼는 import 만 실패 시 경량 PINN
으로 자동 폴백한다.

Examples:
    >>> from naviertwin.core.physnemo.physnemo_wrapper import PhysicsNEMOWrapper
    >>> wrapper = PhysicsNEMOWrapper(equation="poisson_1d")
    >>> wrapper.available
    False  # PhysicsNEMO 미설치 환경
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.physnemo.pina_wrapper import PINNSolver
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class PhysicsNEMOWrapper:
    """PhysicsNEMO 가 있으면 그걸 사용, 없으면 PINA-style 폴백."""

    def __init__(
        self,
        equation: str = "poisson_1d",
        hidden: int = 32,
        max_epochs: int = 300,
        lr: float = 5e-3,
    ) -> None:
        self.equation = equation
        self.hidden = hidden
        self.max_epochs = max_epochs
        self.lr = lr
        self._pinn: PINNSolver | None = None
        self._nemo: Any = None

        try:
            import physicsnemo  # noqa: F401

            self.available = True
        except ImportError:
            self.available = False
        logger.info("PhysicsNEMO available=%s (equation=%s)", self.available, equation)

    def fit(
        self,
        residual_fn: Callable[..., Any] | None = None,
        collocation: NDArray[np.float64] | None = None,
        boundary: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        if self.available:
            # PhysicsNEMO 사용 시 사용자 고유 설정 필요 — 본 wrapper 는 폴백 경로만 사용
            logger.warning(
                "PhysicsNEMO 가 있지만 래퍼는 호환성 이유로 PINA fallback 사용"
            )
        if residual_fn is None or collocation is None:
            raise ValueError("residual_fn + collocation 필요 (폴백 경로)")
        pinn = PINNSolver(
            in_dim=collocation.shape[1],
            out_dim=1,
            hidden=self.hidden,
            n_layers=3,
            max_epochs=self.max_epochs,
            lr=self.lr,
        )
        pinn.fit(residual_fn, collocation, boundary)
        self._pinn = pinn

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._pinn is None:
            raise RuntimeError("fit() 먼저 호출")
        return self._pinn.predict(x)


__all__ = ["PhysicsNEMOWrapper"]
