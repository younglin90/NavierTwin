"""Deep Ritz method — 에너지 함수 최소화 기반 PINN.

포아송 문제 -Δu = f (Ω) + BC 의 경우:
    J[u] = ∫_Ω (1/2 |∇u|² - f·u) dx

샘플 포인트에서 MC 평균으로 에너지 추정 후 최소화.

Examples:
    >>> import numpy as np
    >>> import torch
    >>> from naviertwin.core.physnemo.deep_ritz import DeepRitzSolver
    >>>
    >>> # -u'' = π² sin(π x), u(0)=u(1)=0 → u = sin(π x)
    >>> def f(x):
    ...     return (np.pi ** 2) * torch.sin(np.pi * x)
    >>> solver = DeepRitzSolver(in_dim=1, hidden=32, n_layers=3, max_epochs=500)
    >>> solver.fit(f, domain_range=(0.0, 1.0),
    ...            boundary_x=np.array([[0.0], [1.0]]),
    ...            boundary_u=np.array([[0.0], [0.0]]))
    >>> u_mid = float(solver.predict(np.array([[0.5]]))[0, 0])
    >>> abs(u_mid - 1.0) < 0.3
    True
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class DeepRitzSolver:
    """1D Poisson Deep Ritz — min ∫(1/2 |u'|² - f·u) + λ·BC."""

    def __init__(
        self,
        in_dim: int = 1,
        hidden: int = 32,
        n_layers: int = 3,
        n_collocation: int = 128,
        max_epochs: int = 500,
        lr: float = 5e-3,
        w_bc: float = 10.0,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_collocation = n_collocation
        self.max_epochs = max_epochs
        self.lr = lr
        self.w_bc = w_bc
        self.device = device
        self.seed = seed
        self._model: Any = None
        self._device: Any = None
        self.is_fitted: bool = False
        self.train_losses_: list[float] = []

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self) -> Any:
        import torch
        import torch.nn as nn

        if self.seed is not None:
            torch.manual_seed(self.seed)

        layers: list[nn.Module] = [nn.Linear(self.in_dim, self.hidden), nn.Tanh()]
        layer_idx = 0
        while layer_idx < self.n_layers - 1:
            layers.extend([nn.Linear(self.hidden, self.hidden), nn.Tanh()])
            layer_idx += 1
        layers.append(nn.Linear(self.hidden, 1))
        return nn.Sequential(*layers)

    def fit(
        self,
        f_fn: Callable[..., Any],
        domain_range: tuple[float, float],
        boundary_x: NDArray[np.float64],
        boundary_u: NDArray[np.float64],
    ) -> None:
        """Args:
            f_fn: (x_tensor) → f(x) tensor.
            domain_range: (a, b).
            boundary_x: (N_bc, 1).
            boundary_u: (N_bc, 1).
        """
        import torch

        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        a, b = domain_range
        bc_x = torch.tensor(
            np.asarray(boundary_x, dtype=np.float32), device=self._device
        )
        bc_u = torch.tensor(
            np.asarray(boundary_u, dtype=np.float32), device=self._device
        )

        self.train_losses_ = []
        epoch_idx = 0
        while epoch_idx < self.max_epochs:
            # 샘플링 collocation
            x = torch.rand(self.n_collocation, self.in_dim, device=self._device) * (b - a) + a
            x.requires_grad_(True)
            u = self._model(x)
            du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

            # 에너지: 0.5 |∇u|² - f·u
            f_val = f_fn(x)
            energy = (0.5 * (du ** 2).sum(dim=-1, keepdim=True) - f_val * u).mean()

            # BC
            u_bc = self._model(bc_x)
            bc_loss = torch.nn.functional.mse_loss(u_bc, bc_u)

            loss = energy + self.w_bc * bc_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            self.train_losses_.append(float(loss.item()))
            epoch_idx += 1

        self.is_fitted = True
        logger.info(
            "DeepRitz 학습 완료: final loss=%.6g", self.train_losses_[-1],
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        import torch

        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        x_arr = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            u = self._model(torch.tensor(x_arr, device=self._device))
        return u.cpu().numpy()


__all__ = ["DeepRitzSolver"]
