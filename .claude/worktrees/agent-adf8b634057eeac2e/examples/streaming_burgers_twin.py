"""Burgers 실시간 Digital Twin 데모.

실행:
    python3 examples/streaming_burgers_twin.py

1) 실제 "진실" Burgers 시뮬레이션 (고해상도)
2) 앙상블 초기화 (노이즈 섞인 초기상태)
3) 각 스텝마다 저해상도 모델 forecast + 희소 관측 동화
4) 최종 estimate vs truth 비교
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    from naviertwin.core.digital_twin.streaming_twin import StreamingDigitalTwin
    from naviertwin.core.solver_interfaces.pde_solvers import solve_burgers_1d

    rng = np.random.default_rng(0)
    N = 32
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)

    # 1) 진실 궤적
    u_true0 = np.sin(x) + 0.3 * np.sin(2 * x)
    _, U_true = solve_burgers_1d(u_true0, nu=0.03, L=L, T=0.5, n_steps=200)
    # 매 10 스텝마다 관측
    obs_steps = list(range(10, 200, 20))
    # 공간적으로 균일 간격 8 개 센서
    sensor_idx = np.linspace(0, N - 1, 8, dtype=int)
    H = np.zeros((8, N))
    for i, j in enumerate(sensor_idx):
        H[i, j] = 1.0
    R = 0.01 * np.eye(8)

    # 2) 예측 모델: 저해상도 Burgers "1 step 이동"
    dt = 0.5 / 200

    def model_fn(u: np.ndarray) -> np.ndarray:
        # upwind + diffusion 한 스텝
        nu = 0.03
        dx = L / N
        du_plus = u - np.roll(u, 1)
        du_minus = np.roll(u, -1) - u
        adv = np.where(u > 0, u * du_plus / dx, u * du_minus / dx)
        diff = nu * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2
        return u + dt * (-adv + diff)

    # 3) 앙상블 초기화 — 진실 ±노이즈
    N_ens = 40
    ens0 = U_true[0] + 0.2 * rng.standard_normal((N_ens, N))

    twin = StreamingDigitalTwin(
        state_dim=N, n_ensemble=N_ens,
        model_fn=model_fn, H=H, R=R,
        process_noise=0.005, rng=rng,
    )
    twin.initialize(ens0)

    # 4) 스텝 진행
    errs = []
    for t in range(200):
        twin.step()
        if t in obs_steps:
            y = H @ U_true[t] + 0.05 * rng.standard_normal(8)
            twin.assimilate(y)
        est = twin.estimate()
        err = float(np.linalg.norm(est - U_true[t]) / np.linalg.norm(U_true[t]))
        errs.append(err)

    print(f"[진실 vs 추정] 초기 rel.L2 = {errs[0]:.4f}")
    print(f"[진실 vs 추정] 최종 rel.L2 = {errs[-1]:.4f}")
    print(f"[진실 vs 추정] 평균 rel.L2 = {np.mean(errs):.4f}")
    print(f"[불확실성] 최종 std max = {float(twin.uncertainty().max()):.4f}")

    print()
    print("=" * 50)
    print(f"StreamingTwin Burgers 완료: 추정 오차 {errs[-1]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
