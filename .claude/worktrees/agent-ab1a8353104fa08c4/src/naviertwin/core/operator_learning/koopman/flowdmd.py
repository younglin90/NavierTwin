"""FlowDMD — Coupling Flow INN + DMD.

IKNO 와 유사한 RealNVP 인코더로 얻은 잠재 상태에 standard DMD 를 적용.
잠재 표현에서 선형 스펙트럼을 추출하므로 해석성 + 안정 장기 예측.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.koopman.flowdmd import FlowDMD
    >>> rng = np.random.default_rng(0)
    >>> seqs = rng.standard_normal((3, 20, 4)).astype(np.float32)
    >>> m = FlowDMD(n_features=4, n_blocks=2, hidden=16, dmd_rank=3, max_epochs=3)
    >>> m.fit({"sequences": seqs})
    >>> m.predict(seqs[0, 0], n_steps=5).shape
    (5, 4)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import svd as _svd

from naviertwin.core.operator_learning.koopman.ikno import IKNO
from naviertwin.core.time_series.base import BaseTimeSeries
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class FlowDMD(BaseTimeSeries):
    """IKNO 인코더 + DMD 잠재 예측."""

    def __init__(
        self,
        n_features: int,
        n_blocks: int = 4,
        hidden: int = 32,
        dmd_rank: int | None = None,
        max_epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.hidden = hidden
        self.dmd_rank = dmd_rank
        self.lookback = 1
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.seed = seed

        self._ikno: IKNO | None = None
        self._A_lat: np.ndarray | None = None  # DMD 잠재 선형 진화 행렬

    def _fit_dmd_latent(self, Z_seq: np.ndarray) -> None:
        """잠재 시계열 Z_seq (T, d_lat) 에서 A_lat 를 DMD 로 추정."""
        if Z_seq.shape[0] < 2:
            raise ValueError("잠재 시계열 길이가 2 이상이어야")
        X = Z_seq[:-1].T  # (d_lat, T-1)
        Y = Z_seq[1:].T
        r = self.dmd_rank or min(X.shape[0], X.shape[1])
        U, s, Vt = _svd(X, full_matrices=False)
        U_r = U[:, :r]
        S_r = np.diag(s[:r])
        V_r = Vt[:r].T
        A_tilde = U_r.T @ Y @ V_r @ np.linalg.pinv(S_r)
        # 원 차원으로 재투영
        self._A_lat = U_r @ A_tilde @ U_r.T

    def fit(self, dataset: dict[str, Any]) -> None:
        seqs = np.asarray(dataset["sequences"], dtype=np.float32)
        if seqs.ndim != 3:
            raise ValueError("sequences (N, T, F) 3D 필요")

        # 1) IKNO 학습
        self._ikno = IKNO(
            n_features=self.n_features,
            n_blocks=self.n_blocks,
            hidden=self.hidden,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device,
            seed=self.seed,
        )
        self._ikno.fit({"sequences": seqs})

        # 2) 잠재 시퀀스 얻기 — 첫 번째 시계열에서 DMD
        import torch

        with torch.no_grad():
            z0 = self._ikno._encode(
                torch.tensor(seqs[0], device=self._ikno._device)
            ).cpu().numpy()
        self._fit_dmd_latent(z0)

        self.is_fitted = True
        logger.info(
            "FlowDMD 학습 완료: dmd_rank=%s", self.dmd_rank
        )

    def predict(self, initial_state: np.ndarray, n_steps: int) -> np.ndarray:
        import torch

        self._check_fitted()
        if n_steps < 1:
            raise ValueError("n_steps 는 1 이상")
        x0 = np.asarray(initial_state, dtype=np.float32).ravel()
        with torch.no_grad():
            z = self._ikno._encode(
                torch.tensor(x0[None, :], device=self._ikno._device)
            ).cpu().numpy()[0]

        preds: list[np.ndarray] = []
        step = 0
        while step < n_steps:
            z = self._A_lat @ z
            with torch.no_grad():
                x = self._ikno._decode(
                    torch.tensor(z[None, :], dtype=torch.float32, device=self._ikno._device)
                ).cpu().numpy()[0]
            preds.append(x.copy())
            step += 1
        return np.stack(preds)

    def eigenvalues(self) -> np.ndarray:
        """학습된 A_lat 의 고유값 (spectral analysis)."""
        if self._A_lat is None:
            raise RuntimeError("fit() 먼저 호출")
        return np.linalg.eigvals(self._A_lat)


__all__ = ["FlowDMD"]
