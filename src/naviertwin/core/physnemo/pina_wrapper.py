"""경량 PINN — PINA 스타일 잔차-기반 학습 (PyTorch 직접 구현).

사용자가 PDE 잔차 콜백 ``residual_fn(coords, u(coords))`` 를 제공하면,
MLP 로 u(x) 를 근사하면서 잔차 평균 제곱과 경계조건 손실을 최소화한다.

Examples:
    >>> import numpy as np
    >>> import torch
    >>> from naviertwin.core.physnemo.pina_wrapper import PINNSolver
    >>>
    >>> # d²u/dx² = -π² sin(π x), u(0)=u(1)=0 → u(x)=sin(π x)
    >>> def residual(model, x):
    ...     x = x.requires_grad_(True)
    ...     u = model(x)
    ...     du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    ...     d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
    ...     return d2u + (np.pi ** 2) * torch.sin(np.pi * x)
    >>>
    >>> pinn = PINNSolver(in_dim=1, out_dim=1, hidden=32, n_layers=3, max_epochs=300)
    >>> bc = {"x": np.array([[0.0], [1.0]]), "u": np.array([[0.0], [0.0]])}
    >>> pinn.fit(residual_fn=residual, collocation=np.linspace(0, 1, 64).reshape(-1, 1),
    ...          boundary=bc)
    >>> u_hat = pinn.predict(np.array([[0.5]]))
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class PINNSolver:
    """MLP + PDE residual 손실 기반 경량 PINN.

    Attributes:
        in_dim: 좌표 차원.
        out_dim: 해 차원 (스칼라면 1).
        hidden: MLP 은닉 폭.
        n_layers: MLP 은닉 층 수.
        activation: "tanh" / "gelu" / "sin" (Fourier feature 아님).
        w_phys, w_bc: 잔차/BC 손실 가중치.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden: int = 32,
        n_layers: int = 3,
        activation: str = "tanh",
        w_phys: float = 1.0,
        w_bc: float = 10.0,
        max_epochs: int = 200,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.activation = activation
        self.w_phys = w_phys
        self.w_bc = w_bc
        self.max_epochs = max_epochs
        self.lr = lr
        self.device = device
        self.seed = seed

        self._model: Any = None
        self._device: Any = None
        self.is_fitted: bool = False
        self.train_losses_: list[float] = []
        self.phys_losses_: list[float] = []
        self.bc_losses_: list[float] = []

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

        acts = {"tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}
        act_cls = acts.get(self.activation.lower(), nn.Tanh)

        layers: list[nn.Module] = [nn.Linear(self.in_dim, self.hidden), act_cls()]
        layer_idx = 0
        while layer_idx < self.n_layers - 1:
            layers.extend([nn.Linear(self.hidden, self.hidden), act_cls()])
            layer_idx += 1
        layers.append(nn.Linear(self.hidden, self.out_dim))
        return nn.Sequential(*layers)

    def fit(
        self,
        residual_fn: Callable[..., Any],
        collocation: NDArray[np.float64],
        boundary: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        """PINN 학습.

        Args:
            residual_fn: callable(model, coords_tensor) → residual tensor.
            collocation: (N, in_dim) 물리 영역 내 샘플.
            boundary: {"x": (Nb, in_dim), "u": (Nb, out_dim)} 경계 데이터. 선택.
        """
        import torch

        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        mse = torch.nn.MSELoss()

        X = torch.tensor(
            np.asarray(collocation, dtype=np.float32), device=self._device
        )

        if boundary is not None:
            bc_x = torch.tensor(
                np.asarray(boundary["x"], dtype=np.float32), device=self._device
            )
            bc_u = torch.tensor(
                np.asarray(boundary["u"], dtype=np.float32), device=self._device
            )
        else:
            bc_x = None
            bc_u = None

        self.train_losses_ = []
        self.phys_losses_ = []
        self.bc_losses_ = []

        epoch_idx = 0
        while epoch_idx < self.max_epochs:
            optim.zero_grad()
            res = residual_fn(self._model, X)
            phys = (res ** 2).mean()
            if bc_x is not None and bc_u is not None:
                u_pred = self._model(bc_x)
                bc = mse(u_pred, bc_u)
            else:
                bc = torch.tensor(0.0, device=self._device)
            loss = self.w_phys * phys + self.w_bc * bc
            loss.backward()
            optim.step()
            self.phys_losses_.append(float(phys.item()))
            self.bc_losses_.append(float(bc.item()))
            self.train_losses_.append(float(loss.item()))
            epoch_idx += 1

        self.is_fitted = True
        logger.info(
            "PINN 학습 완료: phys=%.6g, bc=%.6g",
            self.phys_losses_[-1],
            self.bc_losses_[-1],
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        import torch

        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출하세요")
        x_arr = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            u = self._model(torch.tensor(x_arr, device=self._device))
        return u.cpu().numpy()


__all__ = ["PINNSolver"]
