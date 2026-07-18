"""LNO — Laplace Neural Operator (경량 1D 구현).

주파수 도메인 대신 감쇠 지수를 포함하는 복소 극점 p 기반 스펙트럴:

    y_k = Σ_j σ_jk / (s - p_j)

여기서 s = iω. FNO 와 유사하나 복소 pole/residue 쌍을 학습해 과도(transient)
응답을 더 잘 표현한다. 이 구현은 실수부 + 허수부 학습 가능 pole 로 simplify.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.fno.lno import LNO1D
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((10, 32, 1)).astype(np.float32)
    >>> Y = np.cumsum(X, axis=1).astype(np.float32)  # 적분 = 과도 응답
    >>> op = LNO1D(in_channels=1, out_channels=1, n_poles=4, width=8, max_epochs=2)
    >>> op.fit({"inputs": X, "outputs": Y})
    >>> op.predict({"x": X[:2]}).shape
    (2, 32, 1)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _build_laplace_1d(channels: int, n_poles: int) -> Any:
    import torch
    import torch.nn as nn

    class LaplaceConv1d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scale = 1.0 / (channels ** 0.5)
            # p = -α + i·β (안정 조건 α > 0)
            self.log_alpha = nn.Parameter(scale * torch.randn(n_poles))
            self.beta = nn.Parameter(scale * torch.randn(n_poles))
            # 복소 residue σ (in × out × n_poles × 2[re/im])
            self.sigma_re = nn.Parameter(scale * torch.randn(channels, channels, n_poles))
            self.sigma_im = nn.Parameter(scale * torch.randn(channels, channels, n_poles))
            self.n_poles = n_poles

        def forward(self, x: Any) -> Any:  # (B, C, N)
            N = x.shape[-1]
            # FFT 로 주파수 스펙트럼 얻기
            X = torch.fft.rfft(x, norm="forward")  # (B, C, N//2+1) complex
            Nf = X.size(-1)
            omega = torch.linspace(
                0.0, np.pi, Nf, device=x.device
            )  # (Nf,)
            alpha = torch.exp(self.log_alpha)  # (n_poles,)
            # s = iω → (s - p) = (α) + i(ω - β)
            denom_re = alpha.unsqueeze(1)                      # (n_poles, 1)
            denom_im = omega.unsqueeze(0) - self.beta.unsqueeze(1)  # (n_poles, Nf)
            # 1 / (s - p) — 복소 나눗셈
            mag2 = denom_re ** 2 + denom_im ** 2
            inv_re = denom_re / (mag2 + 1e-12)
            inv_im = -denom_im / (mag2 + 1e-12)      # (n_poles, Nf)

            # σ 와 곱 + 입력 채널 경로
            # σ_complex · (1/(s-p)) → (C_out, C_in, Nf) complex
            # einsum: 'oip, pf, ... → ...'
            w_re = torch.einsum("oip,pf->oifp", self.sigma_re, inv_re) - torch.einsum(
                "oip,pf->oifp", self.sigma_im, inv_im
            )
            w_im = torch.einsum("oip,pf->oifp", self.sigma_re, inv_im) + torch.einsum(
                "oip,pf->oifp", self.sigma_im, inv_re
            )
            # w: (C_out, C_in, Nf, n_poles) — sum over p
            w_re = w_re.sum(dim=-1)  # (C_out, C_in, Nf)
            w_im = w_im.sum(dim=-1)
            W = torch.complex(w_re, w_im)

            Y = torch.einsum("oif,bif->bof", W, X)
            return torch.fft.irfft(Y, n=N, norm="forward")

    return LaplaceConv1d()


class LNO1D(BaseOperator):
    """Laplace Neural Operator 1D."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_poles: int = 8,
        width: int = 16,
        n_layers: int = 2,
        max_epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_poles = n_poles
        self.width = width
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
        in_c, out_c, W, P = (
            self.in_channels, self.out_channels, self.width, self.n_poles,
        )
        n_layers = self.n_layers

        class _LNO(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lift = nn.Linear(in_c, W)
                lap = nn.ModuleList()
                ws = nn.ModuleList()
                layer_idx = 0
                while layer_idx < n_layers:
                    lap.append(_build_laplace_1d(W, P))
                    ws.append(nn.Conv1d(W, W, 1))
                    layer_idx += 1
                self.lap = lap
                self.ws = ws
                self.proj = nn.Sequential(
                    nn.Linear(W, 2 * W), nn.GELU(), nn.Linear(2 * W, out_c)
                )

            def forward(self, x: Any) -> Any:  # (B, N, C_in)
                x = self.lift(x).permute(0, 2, 1)  # (B, W, N)
                layer_idx = 0
                while layer_idx < len(self.lap):
                    x = torch.nn.functional.gelu(
                        self.lap[layer_idx](x) + self.ws[layer_idx](x)
                    )
                    layer_idx += 1
                x = x.permute(0, 2, 1)
                return self.proj(x)

        return _LNO()

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(dataset["inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError(f"(B,N,C) 3D 필요: {X.shape}, {Y.shape}")

        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        mse = torch.nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(Y)),
            batch_size=min(self.batch_size, len(X)),
            shuffle=True,
        )
        self.train_losses_ = []
        epoch_idx = 0
        while epoch_idx < self.max_epochs:
            epoch = 0.0
            batches = iter(loader)
            while True:
                try:
                    xb, yb = next(batches)
                except StopIteration:
                    break
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                pred = self._model(xb)
                loss = mse(pred, yb)
                loss.backward()
                optim.step()
                epoch += float(loss.item()) * xb.shape[0]
            epoch /= max(len(X), 1)
            self.train_losses_.append(epoch)
            epoch_idx += 1

        self.is_fitted = True
        logger.info(
            "LNO1D 학습 완료: n_poles=%d loss=%.6g",
            self.n_poles, self.train_losses_[-1],
        )

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        self._check_fitted()
        x = np.asarray(inputs["x"], dtype=np.float32)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None, ...]
        with torch.no_grad():
            y = self._model(torch.tensor(x, device=self._device)).cpu().numpy()
        return y[0] if squeeze else y


__all__ = ["LNO1D"]
