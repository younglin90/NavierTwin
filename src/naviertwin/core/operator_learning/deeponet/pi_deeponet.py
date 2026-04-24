"""PI-DeepONet (Physics-Informed DeepONet).

표준 DeepONet 에 PDE 잔차 항을 추가해 라벨이 적은 데이터로도 학습.
사용자가 ``residual_fn(model, coords)`` 를 제공하면 총 손실:

    L = MSE(data) + λ_phys · MSE(residual)

data loss 는 observed 출력 (라벨)의 재구성,
physics loss 는 PDE 잔차를 평균 제곱.

Examples:
    >>> # 예: u_t = -u (trivial ODE) — residual = du/dt + u
    >>> import numpy as np
    >>> import torch
    >>> from naviertwin.core.operator_learning.deeponet.pi_deeponet import PIDeepONet
    >>>
    >>> def residual_fn(model, coords):
    ...     coords = coords.clone().detach().requires_grad_(True)
    ...     branch = torch.zeros(1, 8, device=coords.device)  # dummy
    ...     u = model(branch, coords)
    ...     du_dt = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
    ...     return du_dt.squeeze(-1) + u.squeeze(0)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from naviertwin.core.operator_learning.deeponet.deeponet import DeepONet
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class PIDeepONet(DeepONet):
    """물리 잔차 항을 포함한 DeepONet.

    Args:
        physics_weight: λ_phys — 물리 잔차 가중치.
        residual_fn: Callable(model_fn, coords_tensor) → residual tensor.
            model_fn(branch_input, trunk_input) 는 연산자 출력을 반환.
        physics_coords: 물리 잔차를 평가할 좌표 샘플 (n_pts, d).
    """

    def __init__(
        self,
        branch_in: int,
        trunk_in: int,
        physics_weight: float = 1.0,
        residual_fn: Callable[..., Any] | None = None,
        physics_coords: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(branch_in=branch_in, trunk_in=trunk_in, **kwargs)
        self.physics_weight = physics_weight
        self.residual_fn = residual_fn
        self.physics_coords = physics_coords
        self.data_losses_: list[float] = []
        self.phys_losses_: list[float] = []

    def _model_fn(self, branch_input: Any, trunk_input: Any) -> Any:
        """(B, m), (q, d) → (B, q) 텐서 연산자 호출."""
        b = self._branch(branch_input)
        t = self._trunk(trunk_input)
        return b @ t.T + self._bias

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        B = np.asarray(dataset["branch_inputs"], dtype=np.float32)
        T = np.asarray(dataset["trunk_inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if B.ndim != 2 or T.ndim != 2 or Y.ndim != 2:
            raise ValueError("branch/trunk/outputs 2D 필요")

        self._device = self._resolve_device()
        self._build()
        self._branch.to(self._device)
        self._trunk.to(self._device)
        trunk_t = torch.tensor(T, device=self._device)

        if self.physics_coords is not None:
            phys_coords = torch.tensor(
                np.asarray(self.physics_coords, dtype=np.float32),
                device=self._device,
            )
        else:
            phys_coords = None

        params = list(self._branch.parameters()) + list(self._trunk.parameters())
        optim = torch.optim.Adam(params, lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(B), torch.tensor(Y)),
            batch_size=min(self.batch_size, len(B)),
            shuffle=True,
        )

        self.train_losses_ = []
        self.data_losses_ = []
        self.phys_losses_ = []

        for _ in range(self.max_epochs):
            epoch_data = 0.0
            epoch_phys = 0.0
            for bb, yb in loader:
                bb = bb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                trunk_feat = self._trunk(trunk_t)
                pred = self._branch(bb) @ trunk_feat.T + self._bias
                data_loss = loss_fn(pred, yb)

                if self.residual_fn is not None and phys_coords is not None:
                    res = self.residual_fn(self._model_fn, phys_coords)
                    phys_loss = (res ** 2).mean()
                else:
                    phys_loss = torch.tensor(0.0, device=self._device)

                loss = data_loss + self.physics_weight * phys_loss
                loss.backward()
                optim.step()
                epoch_data += float(data_loss.item()) * bb.shape[0]
                epoch_phys += float(phys_loss.item()) * bb.shape[0]

            n = max(len(B), 1)
            self.data_losses_.append(epoch_data / n)
            self.phys_losses_.append(epoch_phys / n)
            self.train_losses_.append(
                self.data_losses_[-1] + self.physics_weight * self.phys_losses_[-1]
            )

        self._trunk_cache = T
        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "PI-DeepONet 학습 완료: data=%.6g, phys=%.6g, λ=%.2g",
            self.data_losses_[-1],
            self.phys_losses_[-1],
            self.physics_weight,
        )


__all__ = ["PIDeepONet"]
