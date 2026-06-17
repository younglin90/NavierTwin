"""Online learning — 새 데이터 유입 시 모델 점진적 업데이트.

- OnlineKriging : 최근 버퍼를 유지하며 주기적으로 re-fit
- OnlineNN     : PyTorch 모델에 소량 배치 업데이트 (partial fit)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.online_learning.online_learning import OnlineKriging
    >>> rng = np.random.default_rng(0)
    >>> X0 = rng.standard_normal((5, 2))
    >>> y0 = X0[:, 0] + X0[:, 1]
    >>> ok = OnlineKriging(buffer_size=50, refit_every=5)
    >>> ok.initialize(X0, y0)
    >>> Xn = rng.standard_normal((6, 2))
    >>> tuple(map(lambda x: ok.update(x, float(x.sum())), Xn))
    (None, None, None, None, None, None)
    >>> pred = ok.predict(np.array([[0.1, 0.2]]))
    >>> pred.shape
    (1,)
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class OnlineKriging:
    """GP surrogate 버퍼 + 주기적 refit."""

    def __init__(
        self,
        buffer_size: int = 100,
        refit_every: int = 5,
    ) -> None:
        self.buffer_size = buffer_size
        self.refit_every = refit_every
        self._X: deque = deque(maxlen=buffer_size)
        self._y: deque = deque(maxlen=buffer_size)
        self._counter: int = 0
        self._gp: Any = None
        self.is_initialized: bool = False

    def _refit(self) -> None:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        X = np.array(self._X)
        y = np.array(self._y)
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self._gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self._gp.fit(X, y)

    def initialize(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._X.extend(np.array(X, dtype=np.float64, copy=True))
        self._y.extend(map(float, y))
        self._refit()
        self.is_initialized = True
        logger.info("OnlineKriging 초기화: n=%d", len(self._X))

    def update(self, x: NDArray[np.float64], y: float) -> None:
        if not self.is_initialized:
            raise RuntimeError("initialize() 먼저 호출")
        self._X.append(np.asarray(x, dtype=np.float64).ravel())
        self._y.append(float(y))
        self._counter += 1
        if self._counter % self.refit_every == 0:
            self._refit()

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_initialized or self._gp is None:
            raise RuntimeError("initialize() 먼저 호출")
        return np.asarray(self._gp.predict(X), dtype=np.float64)

    def predict_with_variance(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """예측 평균과 분산을 반환한다."""
        if not self.is_initialized or self._gp is None:
            raise RuntimeError("initialize() 먼저 호출")
        mean, std = self._gp.predict(np.asarray(X, dtype=np.float64), return_std=True)
        return np.asarray(mean, dtype=np.float64), np.asarray(std**2, dtype=np.float64)

    @property
    def input_dim(self) -> int:
        """현재 버퍼의 입력 차원."""
        if not self._X:
            return 0
        return int(np.asarray(self._X[0]).size)


class OnlineNN:
    """PyTorch MLP partial-fit 래퍼."""

    def __init__(
        self,
        model: Any,
        lr: float = 1e-3,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.lr = lr
        self.device = device
        self._device: Any = None
        self._optim: Any = None
        self.is_initialized: bool = False

    def initialize(self) -> None:
        import torch

        dev = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "auto"
            else torch.device(self.device)
        )
        self._device = dev
        self.model.to(dev)
        self._optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.is_initialized = True

    def update(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        n_steps: int = 1,
    ) -> list[float]:
        if not self.is_initialized:
            self.initialize()
        import torch

        xb = torch.tensor(np.asarray(X, dtype=np.float32), device=self._device)
        yb = torch.tensor(np.asarray(y, dtype=np.float32), device=self._device)
        losses: list[float] = []
        step_idx = 0
        while step_idx < n_steps:
            self._optim.zero_grad()
            pred = self.model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            self._optim.step()
            losses.append(float(loss.item()))
            step_idx += 1
        return losses

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        import torch

        if not self.is_initialized:
            self.initialize()
        with torch.no_grad():
            x = torch.tensor(np.asarray(X, dtype=np.float32), device=self._device)
            return self.model(x).cpu().numpy().astype(np.float64)


__all__ = ["OnlineKriging", "OnlineNN"]
