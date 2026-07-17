"""장애물이 있는 2D 채널 유동 LBM 솔버 (D2Q9, bounce-back).

기존 :mod:`naviertwin.core.solver_interfaces.lbm_d2q9` 와 native ``lbm_step`` 은
**주기 경계 + 장애물 없음**이라 실린더 주위 유동을 풀 수 없다. native 커널은
충돌과 스트리밍이 한 함수에 융합돼 있어 그 사이에 들어가야 하는 bounce-back 을
끼울 자리가 없고, 스트리밍이 주기적이라 출구가 입구로 감긴다. 그래서 여기서는
충돌·반사·스트리밍을 직접 제어한다 (400×160 기준 약 5 ms/step).

물리 검증(``tests/test_lbm_obstacle.py``):
    - Re=20 → 셔딩 없는 정상해 + 후류 재순환 거품
    - Re=120 → 카르만 와열, 스트로할 수 St ≈ 0.2

경계 조건:
    - 입구(x=0): 속도 고정(평형 분포). 셔딩을 촉발하려면 ``perturb`` 로 아주 작은
      정현파를 실어 대칭을 깬다 — 완벽 대칭이면 불안정이 자라는 데 아주 오래
      걸린다(교란 없이 12k 스텝을 돌려도 진폭이 5e-5 에 머문다).
    - 출구(x=-1): 영구배(zero-gradient).
    - 위/아래(y): 주기. 채널 폐색률(장애물 높이/ny)이 크면 스트로할 수가
      문헌값보다 높게 나온다 — 벽 간섭의 알려진 효과다.
    - 장애물: fullway bounce-back (고체 노드는 충돌하지 않고 분포를 반사).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "D2Q9_E",
    "D2Q9_OPP",
    "D2Q9_W",
    "SHAPE_KINDS",
    "shape_mask",
    "solve_obstacle_flow",
]

# D2Q9 격자 속도 / 가중치 / 반대 방향 인덱스
D2Q9_E: NDArray[np.int64] = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
D2Q9_W: NDArray[np.float64] = np.array(
    [4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float64
)
D2Q9_OPP: NDArray[np.int64] = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

SHAPE_KINDS = ("circle", "square", "triangle")


def _equilibrium(
    rho: NDArray[np.float64], ux: NDArray[np.float64], uy: NDArray[np.float64]
) -> NDArray[np.float64]:
    """D2Q9 평형 분포 f_eq (2차 전개) — 참조 구현 (초기화·테스트용).

    시간 루프는 :class:`_Workspace` 의 in-place 버전을 쓴다. 이 함수는 매번 새
    배열을 할당하므로 스텝마다 부르면 안 된다.
    """
    usq = ux * ux + uy * uy
    feq = np.empty((9, *rho.shape), dtype=rho.dtype)
    for k in range(9):
        eu = D2Q9_E[k, 0] * ux + D2Q9_E[k, 1] * uy
        feq[k] = D2Q9_W[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)
    return feq


class _Workspace:
    """버퍼를 미리 잡아두고 스텝을 전부 in-place 로 도는 작업 공간.

    LBM 한 스텝은 계산량이 아니라 **메모리**가 병목이다. 순진하게 쓰면
    ``f - omega*(f - feq)`` 한 줄이 매 스텝 (9, ny, nx) 임시 배열을 두 개씩 새로
    잡는데, 400×160 에서 그 한 줄이 스텝 시간의 60% 를 먹었다(측정값 4.88 ms).
    버퍼를 재사용하면 같은 계산이 4 배 빨라지고(결과는 기계 정밀도까지 동일),
    float32 로 내리면 메모리 통행량이 반이라 다시 2 배 이상 빨라진다.
    """

    def __init__(
        self, ny: int, nx: int, solid: NDArray[np.bool_], dtype: Any = np.float32
    ) -> None:
        def buf(*shape: int) -> NDArray[Any]:
            return np.empty(shape, dtype=dtype)

        self.dtype = dtype
        self.rho, self.ux, self.uy = buf(ny, nx), buf(ny, nx), buf(ny, nx)
        self.usq, self.eu, self.tmp = buf(ny, nx), buf(ny, nx), buf(ny, nx)
        self.feq, self.fcol, self.out = buf(9, ny, nx), buf(9, ny, nx), buf(9, ny, nx)
        self.solid_idx = np.flatnonzero(solid.ravel())
        self.weights = D2Q9_W.astype(dtype)

    def macroscopic(self, f: NDArray[Any]) -> None:
        """rho, ux, uy 를 in-place 로 채운다.

        격자 속도가 0/±1 뿐이라 곱셈 없이 더하고 빼면 된다 — E 배열을 곱하면
        (9, ny, nx) 임시 배열이 두 개 더 생긴다.
        """
        rho, ux, uy = self.rho, self.ux, self.uy
        np.sum(f, axis=0, out=rho)
        np.add(f[1], f[5], out=ux)
        ux += f[8]
        ux -= f[3]
        ux -= f[6]
        ux -= f[7]
        ux /= rho
        np.add(f[2], f[5], out=uy)
        uy += f[6]
        uy -= f[4]
        uy -= f[7]
        uy -= f[8]
        uy /= rho

    def equilibrium_inplace(self) -> None:
        """현재 rho/ux/uy 로 self.feq 를 채운다 (할당 없음)."""
        rho, ux, uy, usq, eu, tmp = (
            self.rho, self.ux, self.uy, self.usq, self.eu, self.tmp,
        )
        np.multiply(ux, ux, out=usq)
        np.multiply(uy, uy, out=tmp)
        usq += tmp
        usq *= -1.5
        usq += 1.0  # (1 - 1.5u²) 를 미리 합쳐둔다
        for k in range(9):
            ex, ey = float(D2Q9_E[k, 0]), float(D2Q9_E[k, 1])
            np.multiply(ux, ex, out=eu)
            if ey:
                np.multiply(uy, ey, out=tmp)
                eu += tmp
            fk = self.feq[k]
            np.multiply(eu, eu, out=fk)
            fk *= 4.5
            fk += usq
            eu *= 3.0
            fk += eu
            fk *= rho
            fk *= self.weights[k]

    def collide_bounce_stream(self, f: NDArray[Any], omega: float) -> NDArray[Any]:
        """충돌 → 고체 반사 → 스트리밍. 반환 배열은 재사용 버퍼다."""
        feq, fcol, out, idx = self.feq, self.fcol, self.out, self.solid_idx
        # (1-omega)·f + omega·feq  ≡  f - omega·(f - feq) — 임시 배열 없이.
        np.multiply(f, 1.0 - omega, out=fcol)
        feq *= omega
        fcol += feq
        # 고체 노드는 충돌하지 않고 분포를 반사한다 (fullway bounce-back).
        fcol.reshape(9, -1)[:, idx] = f.reshape(9, -1)[D2Q9_OPP][:, idx]
        for k in range(9):
            dy, dx = int(D2Q9_E[k, 1]), int(D2Q9_E[k, 0])
            if dx or dy:
                out[k] = np.roll(fcol[k], (dy, dx), axis=(0, 1))
            else:
                out[k] = fcol[k]
        return out


def shape_mask(
    nx: int,
    ny: int,
    *,
    kind: str = "circle",
    size: int = 24,
    center: tuple[int, int] | None = None,
) -> NDArray[np.bool_]:
    """장애물 고체 노드 마스크 ``(ny, nx)`` 를 만든다.

    ``size`` 는 **정면 높이**(유동을 막는 y 방향 크기)로 통일한다 — 모양이 달라도
    같은 ``size`` 면 같은 폐색률이라 케이스끼리 비교가 된다.

    Args:
        nx/ny: 격자 크기.
        kind: ``"circle"`` | ``"square"`` | ``"triangle"`` (정삼각형, 꼭짓점이 상류).
        size: 정면 높이 (격자 단위).
        center: 중심 ``(cx, cy)``. 기본은 ``(nx//4, ny//2)``.

    Returns:
        고체면 True 인 ``(ny, nx)`` bool 배열.

    Raises:
        ValueError: 지원하지 않는 ``kind`` 이거나 ``size`` 가 격자를 벗어남.
    """
    if kind not in SHAPE_KINDS:
        raise ValueError(f"지원하지 않는 형상: {kind!r} (가능: {', '.join(SHAPE_KINDS)})")
    size = int(size)
    if size < 3:
        raise ValueError(f"size 는 3 이상이어야 합니다. 현재: {size}")
    if size >= ny:
        raise ValueError(f"size({size}) 가 격자 높이 ny({ny}) 를 넘습니다.")

    cx, cy = center if center is not None else (nx // 4, ny // 2)
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    dx = xx - cx
    dy = yy - cy
    half = size / 2.0

    if kind == "circle":
        return (dx * dx + dy * dy) < half * half
    if kind == "square":
        return (np.abs(dx) < half) & (np.abs(dy) < half)
    # triangle — 꼭짓점이 상류를 향하는 정삼각형(쐐기).
    #   x 가 뒤로 갈수록 폭이 선형으로 벌어지고, size 는 뒷면(정면 높이)이다.
    depth = size * np.sqrt(3.0) / 2.0
    local = dx + depth / 2.0  # 0 = 앞 꼭짓점, depth = 뒷면
    inside_x = (local >= 0.0) & (local <= depth)
    half_width = np.where(depth > 0, local / depth * half, 0.0)
    return inside_x & (np.abs(dy) <= half_width)


def solve_obstacle_flow(
    solid: NDArray[np.bool_],
    *,
    u_in: float = 0.1,
    reynolds: float = 120.0,
    ref_length: float | None = None,
    max_steps: int = 20_000,
    record_every: int = 0,
    record_from: int = 0,
    steady_tol: float = 0.0,
    check_every: int = 200,
    perturb: float = 0.0,
    perturb_steps: int = 0,
    dtype: Any = np.float32,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """장애물 주위 채널 유동을 푼다 (LBM D2Q9 + fullway bounce-back).

    두 가지 모드를 겸한다:

    - **비정상**(카르만 와열): ``record_every`` 로 스냅샷을 모으고, ``perturb`` 로
      셔딩을 촉발한다.
    - **정상**: ``steady_tol`` 을 주면 잔차가 그 아래로 떨어질 때 조기 종료한다.
      Re ≲ 47 에서만 진짜 정상해가 존재한다 — 그 위에서는 수렴하지 않고 셔딩한다.

    Args:
        solid: ``(ny, nx)`` 고체 노드 마스크.
        u_in: 입구 속도 (격자 단위, 압축성 오차를 피하려면 ≲ 0.1).
        reynolds: 레이놀즈 수 = u_in · ref_length / ν.
        ref_length: 기준 길이. 기본은 장애물의 **정면 높이**(마스크에서 계산).
        max_steps: 최대 스텝.
        record_every: >0 이면 이 간격으로 스냅샷 저장 (0 이면 마지막 상태만).
        record_from: 이 스텝 이후부터 저장 (셔딩 발달 구간 버리기).
        steady_tol: >0 이면 ``max|Δux|/u_in < tol`` 에서 조기 종료.
        check_every: 수렴 검사 간격.
        perturb: 입구 정현파 교란 세기 (0 이면 없음).
        perturb_steps: 교란을 유지할 스텝 수 (이후엔 균일류).
        dtype: 계산 정밀도. 기본 ``float32`` — LBM 은 메모리 통행량이 병목이라
            f32 가 f64 보다 2 배 이상 빠르고, 격자 단위 값이 O(0.1~1) 이라 정밀도도
            충분하다(f64 대비 차이 ~1e-7). 검증용으로 ``float64`` 를 줄 수 있다.
        progress_cb: ``(step, total)`` 진행 콜백.

    Returns:
        ``ux``/``uy``/``rho`` (마지막 상태), ``snapshots``(리스트, record_every>0),
        ``steps``, ``converged``, ``residual``, ``omega``, ``nu``, ``ref_length``.

    Raises:
        ValueError: 마스크가 2D 가 아니거나 고체가 전부/전무, 또는 ``omega`` 가
            안정 범위(0, 2)를 벗어남.
    """
    solid = np.asarray(solid, dtype=bool)
    if solid.ndim != 2:
        raise ValueError(f"solid 는 (ny, nx) 2D 여야 합니다. 현재: {solid.shape}")
    if not solid.any():
        raise ValueError("고체 노드가 없습니다 — 장애물 마스크를 확인하세요.")
    if solid.all():
        raise ValueError("전부 고체입니다 — 유체 영역이 없습니다.")
    ny, nx = solid.shape

    if reynolds <= 0.0:
        raise ValueError(f"reynolds 는 양수여야 합니다. 현재: {reynolds}")
    # LBM 은 약압축성이라 마하수 제한이 있다: Ma = u/c_s = u·√3 ≲ 0.3.
    if not 0.0 < u_in <= 0.17:
        raise ValueError(
            f"u_in({u_in}) 이 범위를 벗어납니다 — 격자 단위 속도는 0 < u ≤ 0.17 "
            f"(마하수 제한 Ma=u·√3 ≲ 0.3). 더 빠른 유동은 Re 로 표현하세요."
        )
    if ref_length is None:
        rows = np.flatnonzero(solid.any(axis=1))
        ref_length = float(rows.max() - rows.min() + 1)  # 정면 높이
    ref_length = float(ref_length)
    nu = u_in * ref_length / float(reynolds)
    omega = 1.0 / (3.0 * nu + 0.5)
    # ν>0 이면 omega 는 항상 (0,2) 안이라 "범위 밖" 검사는 의미가 없다. 진짜 위험은
    # omega 가 2 에 **붙는 것**(ν→0, 미해상) — BGK 가 그때 불안정해진다.
    if omega > 1.98:
        raise ValueError(
            f"omega={omega:.4f} 가 2 에 너무 가까워 불안정합니다 (ν={nu:.2e}) — "
            f"장애물 해상도(ref_length={ref_length:.0f} 셀)를 키우거나 "
            f"reynolds({reynolds:.0f}) 를 낮추세요."
        )
    # float32 는 반올림 잡음 때문에 잔차가 ~3e-5 에서 바닥을 친다. 그 아래로 tol 을
    # 주면 수렴 판정이 영원히 안 나 max_steps 까지 도는데, 조용히 느려질 뿐이라
    # 알아채기 어렵다 — 미리 막는다.
    if steady_tol > 0.0 and np.dtype(dtype) == np.float32 and steady_tol < 1e-4:
        raise ValueError(
            f"float32 로는 steady_tol={steady_tol:.1e} 을 판정할 수 없습니다 "
            "(반올림 잡음이 ~3e-5 에서 잔차 바닥을 만들어 영원히 수렴하지 않습니다). "
            "정상해 수렴이 필요하면 dtype=np.float64 를 쓰세요 — 비정상 해석보다 "
            "2 배 느리지만 수렴을 실제로 판정할 수 있습니다."
        )

    dt = np.dtype(dtype)
    y = np.arange(ny)
    inlet_perturbed = (u_in * (1.0 + perturb * np.sin(2.0 * np.pi * y / ny))).astype(dt)
    inlet_uniform = np.full(ny, u_in, dtype=dt)

    ws = _Workspace(ny, nx, solid, dtype=dt)
    rho0 = np.ones((ny, nx), dtype=dt)
    ux0 = np.full((ny, nx), u_in, dtype=dt)
    uy0 = np.zeros((ny, nx), dtype=dt)
    ux0[solid] = 0.0
    f = _equilibrium(rho0, ux0, uy0).astype(dt)

    snapshots: list[dict[str, NDArray[np.float64]]] = []
    prev_ux = ux0.copy()
    residual = float("inf")
    converged = False
    step = 0
    solid_idx = ws.solid_idx

    for step in range(max_steps):
        ws.macroscopic(f)
        rho, ux, uy = ws.rho, ws.ux, ws.uy

        # 입구: 속도 고정(+ 초기 교란). 출구는 아래에서 영구배.
        use_perturb = perturb > 0.0 and step < perturb_steps
        ux[:, 0] = inlet_perturbed if use_perturb else inlet_uniform
        uy[:, 0] = 0.0
        rho[:, 0] = 1.0
        ux.reshape(-1)[solid_idx] = 0.0
        uy.reshape(-1)[solid_idx] = 0.0

        ws.equilibrium_inplace()
        f[:, :, 0] = ws.feq[:, :, 0]

        # collide_bounce_stream 은 feq 를 파괴하므로 이 뒤에서 feq 를 쓰면 안 된다.
        streamed = ws.collide_bounce_stream(f, omega)
        f[...] = streamed
        f[:, :, -1] = f[:, :, -2]  # 출구: 영구배

        if record_every > 0 and step >= record_from and step % record_every == 0:
            snapshots.append(
                {
                    "ux": ux.astype(np.float64),
                    "uy": uy.astype(np.float64),
                    "rho": rho.astype(np.float64),
                }
            )
        if progress_cb is not None and step % max(1, max_steps // 100) == 0:
            progress_cb(step, max_steps)
        if steady_tol > 0.0 and step > 0 and step % check_every == 0:
            residual = float(np.abs(ux - prev_ux).max()) / max(abs(u_in), 1e-12)
            prev_ux[...] = ux
            if residual < steady_tol:
                converged = True
                break
    ux, uy, rho = ws.ux, ws.uy, ws.rho

    if not np.isfinite(ux).all():
        raise ValueError(
            f"해가 발산했습니다 (omega={omega:.3f}) — u_in 을 낮추거나 해상도를 높이세요."
        )
    return {
        "ux": ux,
        "uy": uy,
        "rho": rho,
        "solid": solid,
        "snapshots": snapshots,
        "steps": step + 1,
        "converged": converged,
        "residual": residual,
        "omega": float(omega),
        "nu": float(nu),
        "ref_length": ref_length,
        "u_in": float(u_in),
        "reynolds": float(reynolds),
    }
