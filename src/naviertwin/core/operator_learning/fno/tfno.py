"""TFNO (Tucker-factorized Fourier Neural Operator) — 파라미터 효율 FNO.

SpectralConv2d 의 (C_in × C_out × modes1 × modes2) 복소 가중치 텐서를
Tucker 분해의 경량 근사로 대체한다:

    W[i,o,k1,k2] ≈ Σ_{a,b,c,d} G[a,b,c,d] · Ui[i,a] · Uo[o,b] · U1[k1,c] · U2[k2,d]

rank (a,b,c,d) 를 작게 잡으면 파라미터 수가 C_in·C_out·M1·M2 → rank_core +
부수 인자로 극적으로 줄어든다. 복소수 파라미터는 실수·허수 두 쌍으로 분해.

References:
    Kossaifi et al., "Multi-Grid Tensorized Fourier Neural Operator",
    TMLR 2024 (TFNO).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.fno.tfno import TFNO2D
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((8, 16, 16, 1)).astype(np.float32)
    >>> Y = X ** 2
    >>> op = TFNO2D(in_channels=1, out_channels=1, modes1=4, modes2=4,
    ...             width=8, rank=4, n_layers=2, max_epochs=2)
    >>> op.fit({"inputs": X, "outputs": Y})
    >>> op.predict({"x": X[:2]}).shape
    (2, 16, 16, 1)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _build_tucker_spectral_conv_2d(
    in_c: int, out_c: int, modes1: int, modes2: int, rank: int
) -> Any:
    import torch
    import torch.nn as nn

    class TuckerSpectralConv2d(nn.Module):
        """Tucker-compressed 복소 스펙트럴 가중치."""

        def __init__(self) -> None:
            super().__init__()
            self.modes1 = modes1
            self.modes2 = modes2
            self.out_c = out_c
            r = rank
            scale = 1.0 / (in_c * out_c) ** 0.5

            # 실수/허수 분리 Tucker 코어 + 4 인자
            def _new(shape: tuple[int, ...]) -> Any:
                return nn.Parameter(scale * torch.randn(*shape))

            # 두 "코너" 블록 (FNO 2D 스타일) × 실/허수 = 4 쌍
            for name in ("A", "B"):  # A=[:m1,:m2], B=[-m1:,:m2]
                for part in ("r", "i"):
                    setattr(self, f"core_{name}_{part}", _new((r, r, r, r)))
                    setattr(self, f"Ui_{name}_{part}", _new((in_c, r)))
                    setattr(self, f"Uo_{name}_{part}", _new((out_c, r)))
                    setattr(self, f"U1_{name}_{part}", _new((modes1, r)))
                    setattr(self, f"U2_{name}_{part}", _new((modes2, r)))

        def _reassemble(self, name: str) -> Any:
            """Tucker 인자 → 복소 텐서 (in_c, out_c, modes1, modes2)."""
            r = getattr(self, f"core_{name}_r")
            i = getattr(self, f"core_{name}_i")
            Ui_r = getattr(self, f"Ui_{name}_r")
            Ui_i = getattr(self, f"Ui_{name}_i")
            Uo_r = getattr(self, f"Uo_{name}_r")
            Uo_i = getattr(self, f"Uo_{name}_i")
            U1_r = getattr(self, f"U1_{name}_r")
            U1_i = getattr(self, f"U1_{name}_i")
            U2_r = getattr(self, f"U2_{name}_r")
            U2_i = getattr(self, f"U2_{name}_i")

            # 실수부/허수부 각각 Tucker 재조립
            real = torch.einsum("abcd,ia,ob,kc,ld->iokl", r, Ui_r, Uo_r, U1_r, U2_r)
            imag = torch.einsum("abcd,ia,ob,kc,ld->iokl", i, Ui_i, Uo_i, U1_i, U2_i)
            return torch.complex(real, imag)

        def forward(self, x: Any) -> Any:  # (B, C_in, H, W)
            B, _, H, W = x.shape
            x_ft = torch.fft.rfft2(x, norm="forward")
            out_ft = torch.zeros(
                B, self.out_c, H, W // 2 + 1,
                dtype=torch.cfloat, device=x.device,
            )
            m1 = min(self.modes1, H)
            m2 = min(self.modes2, W // 2 + 1)
            wA = self._reassemble("A")[:, :, :m1, :m2]
            wB = self._reassemble("B")[:, :, :m1, :m2]
            out_ft[:, :, :m1, :m2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], wA
            )
            out_ft[:, :, -m1:, :m2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], wB
            )
            return torch.fft.irfft2(out_ft, s=(H, W), norm="forward")

    return TuckerSpectralConv2d()


class TFNO2D(BaseOperator):
    """Tucker-factorized 2D FNO.

    Attributes:
        rank: Tucker 코어 차원. 작을수록 파라미터 수 급감.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes1: int = 8,
        modes2: int = 8,
        width: int = 16,
        rank: int = 4,
        n_layers: int = 4,
        max_epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.rank = rank
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self._model: Any = None
        self._device: Any = None
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

        in_c, out_c, W, M1, M2, R = (
            self.in_channels,
            self.out_channels,
            self.width,
            self.modes1,
            self.modes2,
            self.rank,
        )
        n_layers = self.n_layers

        class _TFNO2D(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lift = nn.Linear(in_c, W)
                self.specs = nn.ModuleList(
                    [
                        _build_tucker_spectral_conv_2d(W, W, M1, M2, R)
                        for _ in range(n_layers)
                    ]
                )
                self.ws = nn.ModuleList(
                    [nn.Conv2d(W, W, 1) for _ in range(n_layers)]
                )
                self.proj1 = nn.Linear(W, 4 * W)
                self.proj2 = nn.Linear(4 * W, out_c)

            def forward(self, x: Any) -> Any:
                x = self.lift(x).permute(0, 3, 1, 2)
                for sp, w in zip(self.specs, self.ws):
                    x = torch.nn.functional.gelu(sp(x) + w(x))
                x = x.permute(0, 2, 3, 1)
                x = torch.nn.functional.gelu(self.proj1(x))
                return self.proj2(x)

        return _TFNO2D()

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(dataset["inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if X.ndim != 4 or Y.ndim != 4:
            raise ValueError(f"(B,H,W,C) 4D 필요: {X.shape}, {Y.shape}")

        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(Y)),
            batch_size=min(self.batch_size, len(X)),
            shuffle=True,
        )
        self.train_losses_ = []
        for _ in range(self.max_epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_loss /= max(len(X), 1)
            self.train_losses_.append(epoch_loss)

        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "TFNO2D 학습 완료: rank=%d, loss=%.6g",
            self.rank,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        self._check_fitted()
        x = np.asarray(inputs["x"], dtype=np.float32)
        squeeze = x.ndim == 3
        if squeeze:
            x = x[None, ...]
        with torch.no_grad():
            y = self._model(torch.tensor(x, device=self._device)).cpu().numpy()
        return y[0] if squeeze else y

    def param_count(self) -> int:
        """현재 모델 파라미터 총수 (fit 이후에만 정확)."""
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())


__all__ = ["TFNO2D"]
