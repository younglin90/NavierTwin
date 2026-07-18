"""SIMP (Solid Isotropic Material with Penalization) 위상 최적화.

2D 컴플라이언스 최소화 문제를 NumPy 로 간단히 구현한다.
    minimize  c(ρ) = Σ (E(ρ_e) · u_eᵀ k_e u_e)
    s.t.      Σ ρ_e · v_e = V* (volume fraction)
              0 < ρ_min ≤ ρ_e ≤ 1

OC (Optimality Criteria) 업데이트 + 간단한 FEA (unit element stiffness).
실제 CFD 위상 최적화는 PyTopo3D/DL4TO 에 위임하지만, 이 모듈은 교육용 MVP.

Examples:
    >>> from naviertwin.core.optimization.topology_opt import simp_2d
    >>> rho = simp_2d(nx=10, ny=6, vol_frac=0.5, penal=3.0, n_iter=20)
    >>> rho.shape
    (6, 10)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import solve as _solve
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _element_stiffness() -> NDArray[np.float64]:
    """Q4 element stiffness matrix (unit square, E=1, ν=0.3)."""
    E = 1.0
    nu = 0.3
    k = np.array(
        [
            1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8,
        ]
    )
    KE = (E / (1 - nu ** 2)) * np.array(
        [
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ]
    )
    return KE


def simp_2d(
    nx: int = 30,
    ny: int = 20,
    vol_frac: float = 0.5,
    penal: float = 3.0,
    rho_min: float = 1e-3,
    n_iter: int = 50,
    load_node: int | None = None,
    fixed_left_edge: bool = True,
) -> NDArray[np.float64]:
    """MBB-beam 스타일 2D 컴플라이언스 최소화.

    좌측 모서리 고정, 상단 중앙(load_node) 에 단위 하중.

    Args:
        nx, ny: 격자 크기.
        vol_frac: 허용 체적 분율.
        penal: SIMP penalty.
        rho_min: 최소 밀도.
        n_iter: OC 반복 수.
        load_node: 하중 위치(dof index). None 이면 상단 중앙.
        fixed_left_edge: 좌측 x=0 모든 노드 고정.

    Returns:
        밀도 분포 (ny, nx).
    """
    n_el = nx * ny
    n_node = (nx + 1) * (ny + 1)
    n_dof = 2 * n_node

    def _node(i: int, j: int) -> int:
        return j * (nx + 1) + i

    # 글로벌 요소별 dof 인덱스
    nodes = np.arange(n_node, dtype=int).reshape(ny + 1, nx + 1)
    n1 = nodes[:-1, :-1].ravel()
    n2 = nodes[:-1, 1:].ravel()
    n3 = nodes[1:, 1:].ravel()
    n4 = nodes[1:, :-1].ravel()
    edof = np.column_stack(
        [
            2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1,
            2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1,
        ]
    )

    KE = _element_stiffness()

    # 고정 DOF
    if fixed_left_edge:
        left_nodes = nodes[:, 0]
        fixed_arr = np.ravel(np.column_stack([2 * left_nodes, 2 * left_nodes + 1]))
    else:
        fixed_arr = np.array([], dtype=int)
    free_arr = np.setdiff1d(np.arange(n_dof), fixed_arr)

    # 하중 벡터
    F = np.zeros(n_dof)
    if load_node is None:
        load_node = 2 * _node(nx, ny // 2) + 1  # 우측 중앙 y 방향
    F[load_node] = -1.0

    rho = np.full((ny, nx), vol_frac)
    it = 0
    while it < n_iter:
        # 글로벌 K 조립
        K = np.zeros((n_dof, n_dof))
        rho_flat = rho.ravel()
        elem_modulus = rho_min + (1 - rho_min) * rho_flat ** penal
        np.add.at(
            K,
            (edof[:, :, None], edof[:, None, :]),
            elem_modulus[:, None, None] * KE[None, :, :],
        )

        # 선형 시스템 K u = F (free DOFs 만)
        try:
            u_free = _solve(K[np.ix_(free_arr, free_arr)], F[free_arr])
        except np.linalg.LinAlgError:
            logger.warning("SIMP: 특이 행렬 — 반복 중단")
            break
        U = np.zeros(n_dof)
        U[free_arr] = u_free

        # 요소별 컴플라이언스 dc/drho
        Ue = U[edof]
        ke_vals = np.einsum("ei,ij,ej->e", Ue, KE, Ue)
        c = float(np.sum(elem_modulus * ke_vals))
        dc = (
            -penal * (1 - rho_min) * rho_flat ** (penal - 1) * ke_vals
        ).reshape(ny, nx)

        # OC 업데이트 (bisection on Lagrange multiplier)
        l1, l2 = 1e-9, 1e9
        move = 0.2
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            rho_new = np.maximum(
                rho_min,
                np.maximum(
                    rho - move,
                    np.minimum(
                        1.0,
                        np.minimum(
                            rho + move,
                            rho * np.sqrt(-dc / lmid),
                        ),
                    ),
                ),
            )
            if rho_new.sum() - vol_frac * n_el > 0:
                l1 = lmid
            else:
                l2 = lmid
        rho = rho_new

        if it % max(1, n_iter // 5) == 0:
            logger.debug("SIMP iter %d: c=%.4g", it, c)
        it += 1

    return rho


__all__ = ["simp_2d"]
