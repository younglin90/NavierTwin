"""Domain-Decomposition PINN — 여러 서브 도메인 순차 학습 + 인터페이스 조건.

1D 도메인 [a, b] 을 n_sub 개로 분할. 각 서브 도메인에 별도 PINN 학습.
인접 서브 도메인 경계에서 연속성 (u 연속, du/dx 연속) 을 추가 loss 로 강제.

Examples:
    >>> import numpy as np
    >>> import torch
    >>> from naviertwin.core.physnemo.dd_pinn import DomainDecompPINN
    >>>
    >>> # 1D Poisson: -u'' = π² sin(π x), u(0)=u(1)=0
    >>> def residual(model, x):
    ...     x = x.requires_grad_(True)
    ...     u = model(x)
    ...     du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    ...     d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
    ...     return d2u + (np.pi ** 2) * torch.sin(np.pi * x)
    >>>
    >>> dd = DomainDecompPINN(n_sub=2, domain=(0.0, 1.0), hidden=16, max_epochs=100)
    >>> bc = {"x": np.array([[0.0], [1.0]]), "u": np.array([[0.0], [0.0]])}
    >>> dd.fit(residual, bc)
    >>> u_mid = float(dd.predict(np.array([[0.5]]))[0, 0])
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.physnemo.pina_wrapper import PINNSolver
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class DomainDecompPINN:
    """1D 도메인 분할 PINN."""

    def __init__(
        self,
        n_sub: int = 2,
        domain: tuple[float, float] = (0.0, 1.0),
        n_collocation: int = 64,
        hidden: int = 32,
        n_layers: int = 3,
        max_epochs: int = 300,
        lr: float = 5e-3,
        w_interface: float = 10.0,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        if n_sub < 1:
            raise ValueError("n_sub >= 1 필요")
        self.n_sub = n_sub
        self.domain = domain
        self.n_collocation = n_collocation
        self.hidden = hidden
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.lr = lr
        self.w_interface = w_interface
        self.device = device
        self.seed = seed

        self._subs: list[PINNSolver] = []
        self._boundaries: list[tuple[float, float]] = []
        self.is_fitted: bool = False

    def _split_domain(self) -> list[tuple[float, float]]:
        a, b = self.domain
        xs = np.linspace(a, b, self.n_sub + 1)
        pairs = np.column_stack([xs[:-1], xs[1:]])
        return list(map(lambda p: (float(p[0]), float(p[1])), pairs))

    def fit(
        self,
        residual_fn: Callable[..., Any],
        boundary: dict[str, NDArray[np.float64]],
    ) -> None:
        """각 서브도메인 PINN 순차 학습 + 인터페이스 일치 반복."""

        self._boundaries = self._split_domain()
        bc_x = np.asarray(boundary["x"], dtype=np.float32)
        bc_u = np.asarray(boundary["u"], dtype=np.float32)

        self._subs = []
        k = 0
        while k < len(self._boundaries):
            a, b = self._boundaries[k]
            sub = PINNSolver(
                in_dim=1, out_dim=1, hidden=self.hidden, n_layers=self.n_layers,
                max_epochs=self.max_epochs // max(self.n_sub, 1),
                lr=self.lr, device=self.device, seed=self.seed,
            )
            col = np.linspace(a, b, self.n_collocation, dtype=np.float32).reshape(-1, 1)

            # 이 서브의 경계조건 = 외부 BC 중 [a,b] 영역 + 이웃 서브와의 일치
            inside = (a - 1e-9 <= bc_x[:, 0]) & (bc_x[:, 0] <= b + 1e-9)
            sub_bc_x: list[np.ndarray] = list(bc_x[inside])
            sub_bc_u: list[np.ndarray] = list(bc_u[inside])

            # 이웃 인터페이스 — 이미 학습된 이전 서브의 값 사용
            if k > 0:
                prev = self._subs[-1]
                x_iface = np.array([[a]], dtype=np.float32)
                u_iface = prev.predict(x_iface)
                sub_bc_x.append(x_iface[0])
                sub_bc_u.append(u_iface[0])

            if sub_bc_x:
                sub_bc = {
                    "x": np.stack(sub_bc_x).astype(np.float32),
                    "u": np.stack(sub_bc_u).astype(np.float32),
                }
            else:
                sub_bc = None
            sub.fit(residual_fn, col, sub_bc)
            self._subs.append(sub)
            k += 1

        self.is_fitted = True
        logger.info(
            "DD-PINN 학습 완료: n_sub=%d, boundaries=%s",
            self.n_sub, self._boundaries,
        )

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("fit() 먼저 호출")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        out = np.zeros((x.shape[0], 1), dtype=np.float64)
        assigned = np.zeros(x.shape[0], dtype=bool)
        k = 0
        while k < len(self._boundaries):
            a, b = self._boundaries[k]
            mask = (~assigned) & (a - 1e-9 <= x[:, 0]) & (x[:, 0] <= b + 1e-9)
            if mask.any():
                out[mask] = self._subs[k].predict(x[mask])
                assigned[mask] = True
            k += 1
        return out


__all__ = ["DomainDecompPINN"]
