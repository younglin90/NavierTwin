"""2D D2Q9 Lattice Boltzmann solver (numpy 직접 구현).

LBGK 충돌 연산자. Moving-wall (Zou-He) 상하벽 처리 등은 생략하고 주기 경계
+ top moving wall 과 비슷한 단순 설정으로 cavity-like 유동을 생성한다.

사용 목적: NavierTwin 의 ROM/operator 학습용 **합성 스냅샷 빠른 생성**.
정확도보다는 재현성/속도 우선.

Examples:
    >>> from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9
    >>> lbm = LBMD2Q9(nx=32, ny=32, tau=0.8, u_top=0.05)
    >>> snapshots = lbm.run(n_steps=200, record_every=20)
    >>> snapshots.shape
    (10, 32, 32, 3)  # (T, ny, nx, [rho, ux, uy])
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

# D2Q9 lattice 속도, 가중치
_CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
_CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
_W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float64)
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


class LBMD2Q9:
    """D2Q9 LBGK — top-lid driven cavity 유사 설정."""

    def __init__(
        self,
        nx: int = 32,
        ny: int = 32,
        tau: float = 0.8,
        u_top: float = 0.05,
    ) -> None:
        if tau <= 0.5:
            raise ValueError("tau > 0.5 필요 (점성 양수)")
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.u_top = u_top

        # 초기화: 평형 분포 (정지)
        self.rho = np.ones((ny, nx), dtype=np.float64)
        self.ux = np.zeros((ny, nx), dtype=np.float64)
        self.uy = np.zeros((ny, nx), dtype=np.float64)
        self.f = self._equilibrium(self.rho, self.ux, self.uy)

    def _equilibrium(
        self,
        rho: NDArray[np.float64],
        ux: NDArray[np.float64],
        uy: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Fully vectorized over 9 velocities via broadcasting
        usqr = ux ** 2 + uy ** 2
        cu = _CX[:, None, None] * ux[None, :, :] + _CY[:, None, None] * uy[None, :, :]
        feq = (
            _W[:, None, None]
            * rho[None, :, :]
            * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * usqr[None, :, :])
        )
        return feq

    def _stream(self) -> None:
        """Streaming step — 주기 경계 (np.roll)."""
        self.f = np.stack(
            (
                self.f[0],
                np.roll(self.f[1], shift=(0, 1), axis=(0, 1)),
                np.roll(self.f[2], shift=(1, 0), axis=(0, 1)),
                np.roll(self.f[3], shift=(0, -1), axis=(0, 1)),
                np.roll(self.f[4], shift=(-1, 0), axis=(0, 1)),
                np.roll(self.f[5], shift=(1, 1), axis=(0, 1)),
                np.roll(self.f[6], shift=(1, -1), axis=(0, 1)),
                np.roll(self.f[7], shift=(-1, -1), axis=(0, 1)),
                np.roll(self.f[8], shift=(-1, 1), axis=(0, 1)),
            ),
            axis=0,
        )

    def _collide(self) -> None:
        """LBGK 충돌 + macroscopic 재계산."""
        rho = self.f.sum(axis=0)
        ux = (self.f * _CX[:, None, None]).sum(axis=0) / np.maximum(rho, 1e-12)
        uy = (self.f * _CY[:, None, None]).sum(axis=0) / np.maximum(rho, 1e-12)

        # top moving-wall (y = 0 가정 — 상단행에서 ux = u_top 강제)
        ux[0, :] = self.u_top
        uy[0, :] = 0.0

        feq = self._equilibrium(rho, ux, uy)
        self.f = self.f - (self.f - feq) / self.tau
        self.rho = rho
        self.ux = ux
        self.uy = uy

    def step(self) -> None:
        self._collide()
        self._stream()

    def run(
        self, n_steps: int = 500, record_every: int = 50
    ) -> NDArray[np.float64]:
        """스냅샷 리스트 반환: shape (n_records, ny, nx, 3) [rho, ux, uy]."""
        snaps: list[NDArray[np.float64]] = []
        t = 1
        while t <= n_steps:
            self.step()
            if t % record_every == 0:
                snap = np.stack([self.rho.copy(), self.ux.copy(), self.uy.copy()], axis=-1)
                snaps.append(snap)
            t += 1
        logger.info(
            "LBM D2Q9 완료: nx=%d ny=%d tau=%.3f 스텝=%d 기록=%d",
            self.nx, self.ny, self.tau, n_steps, len(snaps),
        )
        return np.asarray(snaps)


__all__ = ["LBMD2Q9"]
